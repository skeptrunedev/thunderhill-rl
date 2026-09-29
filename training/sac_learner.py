"""GPU side of the async SAC trainer (sac_async.py): networks, a GPU-resident
replay buffer and one SAC update captured as a single CUDA graph.

The update is Stable-Baselines3 2.7 SAC, step for step (the pilot's algorithm):
entropy coefficient first (with its pre-step value used afterwards), then the
twin critics against the target critics, then the actor against the updated
critics, then Polyak averaging. Speed comes from removing per-kernel launch and
Python overhead, following two references:

- pytorch-labs/LeanRL sac_continuous_action_torchcompile.py: twin critics
  evaluated as one batched network instead of two modules, torch.compile of the
  update, CUDA graph capture on top (their CudaGraphModule), in-place
  target averaging (lerp_) and in-place alpha.
- younggyoseo/FastTD3 train.py / fast_td3_utils.py: the replay buffer lives on
  the GPU and is sampled with on-device random indices, so a whole update, the
  batch sampling included, runs without a host round trip; large batches.

One captured graph replays sampling + all three optimizer steps + target update,
so the Python cost per update is one graph launch.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0  # SB3 SAC actor
TANH_EPSILON = 1e-6  # SB3 TanhBijector


def _uniform_(tensor, fan_in):
    # nn.Linear's default initialization.
    bound = 1.0 / math.sqrt(fan_in)
    return nn.init.uniform_(tensor, -bound, bound)


class Actor(nn.Module):
    """ReLU MLP with mean and log-std heads (SB3 SAC MlpPolicy actor)."""

    def __init__(self, observation_size, action_size, hidden):
        super().__init__()
        sizes = [observation_size, *hidden]
        self.trunk = nn.Sequential(*[
            module for i, o in zip(sizes[:-1], sizes[1:]) for module in (nn.Linear(i, o), nn.ReLU())
        ])
        self.mean = nn.Linear(sizes[-1], action_size)
        self.log_std = nn.Linear(sizes[-1], action_size)

    def sample(self, observation):
        """Reparameterized tanh-Gaussian action and its log-probability."""
        latent = self.trunk(observation)
        mean = self.mean(latent)
        std = self.log_std(latent).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()
        gaussian = mean + std * torch.randn_like(mean)
        action = torch.tanh(gaussian)
        log_prob = (-0.5 * ((gaussian - mean) / std) ** 2 - std.log()
                    - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - torch.log(1 - action ** 2 + TANH_EPSILON).sum(-1)
        return action, log_prob

    def flat_parameters(self):
        """Parameters in the order sac_async.NumpyPolicy reads them."""
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])


class Critics(nn.Module):
    """Twin Q-networks as one stacked network evaluated with batched matmuls."""

    def __init__(self, observation_size, action_size, hidden, count=2):
        super().__init__()
        sizes = [observation_size + action_size, *hidden, 1]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i, o in zip(sizes[:-1], sizes[1:]):
            self.weights.append(nn.Parameter(_uniform_(torch.empty(count, i, o), i)))
            self.biases.append(nn.Parameter(_uniform_(torch.empty(count, 1, o), i)))

    def forward(self, observation, action, detach=False):
        x = torch.cat([observation, action], -1).expand(len(self.weights[0]), -1, -1)
        last = len(self.weights) - 1
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            if detach:
                w, b = w.detach(), b.detach()
            x = torch.baddbmm(b, x, w)
            if k < last:
                x = F.relu(x)
        return x.squeeze(-1)  # (critics, batch)


class ReplayBuffer:
    """Circular GPU buffer of (obs, action, reward, next_obs, terminated)."""

    # prior: 1 where the action taken was a forced demonstration (sac_async
    # --straight-throttle-prior), for the actor's optional demonstration loss.
    FIELDS = ("obs", "action", "reward", "next_obs", "terminated", "prior")

    def __init__(self, capacity, observation_size, action_size, device):
        self.capacity, self.position, self.count = capacity, 0, 0
        shapes = dict(obs=(observation_size,), action=(action_size,), reward=(),
                      next_obs=(observation_size,), terminated=(), prior=())
        self.data = {k: torch.zeros((capacity, *shapes[k]), device=device) for k in self.FIELDS}
        # Read by the captured update, so it is a device tensor updated in place.
        self.size = torch.zeros((), device=device)

    def add(self, rows: dict):
        n = len(rows["reward"])
        if n == 0:
            return
        # Pinned, non-blocking copies queue behind the running updates instead of
        # waiting for them.
        device = lambda array: torch.from_numpy(np.ascontiguousarray(array)).pin_memory().to(  # noqa: E731
            self.size.device, non_blocking=True)
        index = device((self.position + np.arange(n)) % self.capacity)
        rows = dict(rows)
        rows.setdefault("prior", np.zeros(n, np.float32))
        for k in self.FIELDS:
            self.data[k].index_copy_(0, index, device(rows[k]))
        self.position = (self.position + n) % self.capacity
        self.count = min(self.count + n, self.capacity)
        self.size.fill_(self.count)

    def sample(self, batch_size):
        index = (torch.rand(batch_size, device=self.size.device) * self.size).long()
        return {k: v[index] for k, v in self.data.items()}

    def save(self, path: Path):
        # Oldest first, so a reload is a plain append.
        order = (self.position + np.arange(self.count)) % self.capacity if self.count == self.capacity \
            else np.arange(self.count)
        tmp = path.with_suffix(".tmp.npz")
        # Reorder on the host: a GPU gather needs a second copy of the largest field.
        np.savez(tmp, **{k: v.cpu().numpy()[order] for k, v in self.data.items()})
        tmp.replace(path)

    def load(self, path: Path):
        with np.load(path) as saved:
            # Buffers saved before the prior column load with prior = 0.
            rows = {k: saved[k][-self.capacity:] for k in self.FIELDS if k in saved}
        for start in range(0, len(rows["reward"]), 262_144):
            self.add({k: v[start:start + 262_144] for k, v in rows.items()})


class SAC:
    def __init__(self, *, observation_size, action_size, hidden, buffer_size, batch_size,
                 learning_rate, gamma, tau, device="cuda", compile=True, cuda_graph=True,
                 demo_weight=0.0):
        self.device, self.batch_size, self.gamma, self.tau = device, batch_size, gamma, tau
        self.learning_rate = learning_rate
        # demo_weight: squared-error pull of the actor's pedal mean toward the pedal
        # of forced demonstration transitions (buffer prior = 1), DDPGfD-style.
        self.demo_weight = demo_weight
        self.actor = Actor(observation_size, action_size, hidden).to(device)
        self.critic = Critics(observation_size, action_size, hidden).to(device)
        self.critic_target = Critics(observation_size, action_size, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)  # SB3 "auto": init 1.0
        self.target_entropy = -float(action_size)
        # capturable keeps Adam's step counters on the GPU, as graph replay requires.
        adam = dict(lr=learning_rate, capturable=True, foreach=True)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), **adam)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), **adam)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], **adam)
        self.buffer = ReplayBuffer(buffer_size, observation_size, action_size, device)
        self.updates = 0
        wrap = torch.compile if compile else (lambda f: f)
        self._sample, self._critic, self._actor = (
            wrap(f) for f in (self._sample_actor, self._critic_loss, self._actor_loss))
        self._step = self._update
        self._use_graph = cuda_graph
        self._graph = None
        self.metrics = None

    # The update is split into compiled pure functions (forward passes and
    # losses) around eager backward/optimizer calls: a backward inside a compiled
    # region breaks the graph there, and alpha and actor losses share the actor's
    # forward. The CUDA graph then removes the eager calls' launch overhead.
    def _sample_actor(self, obs):
        return self.actor.sample(obs)

    def _critic_loss(self, batch, alpha):
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch["next_obs"])
            next_q = self.critic_target(batch["next_obs"], next_action).min(0).values
            next_q = next_q - alpha * next_log_prob
            target = batch["reward"] + (1 - batch["terminated"]) * self.gamma * next_q
        q = self.critic(batch["obs"], batch["action"])
        return 0.5 * ((q - target) ** 2).mean(-1).sum()

    def _actor_loss(self, obs, action_pi, log_prob, alpha):
        q_pi = self.critic(obs, action_pi, detach=True).min(0).values
        return (alpha * log_prob - q_pi).mean()

    def _demo_loss(self, batch):
        pedal = torch.tanh(self.actor.mean(self.actor.trunk(batch["obs"])))[:, 1]
        error = (pedal - batch["action"][:, 1]) ** 2
        return (batch["prior"] * error).sum() / batch["prior"].sum().clamp(min=1.0)

    def _update(self):
        batch = self.buffer.sample(self.batch_size)
        action_pi, log_prob = self._sample(batch["obs"])

        # Entropy coefficient first; its pre-step value is used below (as SB3).
        alpha = self.log_alpha.detach().exp()
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=False)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        critic_loss = self._critic(batch, alpha)
        self.critic_optimizer.zero_grad(set_to_none=False)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self._actor(batch["obs"], action_pi, log_prob, alpha)
        if self.demo_weight:
            actor_loss = actor_loss + self.demo_weight * self._demo_loss(batch)
        self.actor_optimizer.zero_grad(set_to_none=False)
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            torch._foreach_lerp_(list(self.critic_target.parameters()),
                                 list(self.critic.parameters()), self.tau)
        return torch.stack([critic_loss.detach(), actor_loss.detach(), alpha_loss.detach(),
                            alpha[0], -log_prob.detach().mean()])

    def _capture(self):
        # Warm up on a side stream (compilation, autotuning, optimizer state),
        # then record one update into a graph whose replay reuses its memory.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.metrics = self._step()
        # The warm-up and capture ran real updates on real data (+4).
        self.updates += 4

    def update(self, count=1):
        """Run `count` updates; returns without waiting for the GPU."""
        if not self._use_graph:
            for _ in range(count):
                self.metrics = self._step()
                self.updates += 1
            return
        if self._graph is None:
            self._capture()
            count = max(0, count - 4)
        for _ in range(count):
            self._graph.replay()
        self.updates += count

    def snapshot_actor(self):
        """Start an asynchronous copy of the actor's parameters to pinned host
        memory; returns (host tensor, event, update count) to poll."""
        host = torch.empty(sum(p.numel() for p in self.actor.parameters()), pin_memory=True)
        host.copy_(self.actor.flat_parameters(), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        return host, event, self.updates

    def metric_values(self):
        if self.metrics is None:
            return {}
        values = self.metrics.cpu().tolist()
        return dict(zip(("train/critic_loss", "train/actor_loss", "train/ent_coef_loss",
                         "train/ent_coef", "train/entropy"), values))

    # Persistence -------------------------------------------------------------
    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    critic_target=self.critic_target.state_dict(),
                    log_alpha=self.log_alpha.detach().clone(),
                    actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict(),
                    alpha_optimizer=self.alpha_optimizer.state_dict(),
                    updates=self.updates)

    def load_state_dict(self, state):
        # In-place copies keep the tensors a captured graph (if any) points at.
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
            optimizer = getattr(self, name)
            optimizer.load_state_dict(state[name])
            # The saved param_groups carry the old run's lr; the configured one wins.
            for group in optimizer.param_groups:
                group["lr"] = self.learning_rate
        self.updates = state["updates"]
