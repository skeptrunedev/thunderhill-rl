"""A trained SAC actor as a NumPy teacher: the deterministic action for any observation.

Rev learns to ride by imitation with DAgger: the teacher labels every state, including
the states Rev reaches through its own mistakes, with the action it would take there.
The label is the actor's mean action (tanh of the mean head), not the exploration
samples SAC recorded while training.

Loading reads the sac_async checkpoint once with torch; acting is NumPy only, batched
or one observation at a time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class SacTeacher:
    def __init__(self, checkpoint: Path | str, pedal_gain: float = 1.25):
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["sac"]["actor"]
        trunk = sorted({int(k.split(".")[1]) for k in state if k.startswith("trunk.")})
        self.layers = [(state[f"trunk.{i}.weight"].float().numpy().T, state[f"trunk.{i}.bias"].float().numpy())
                       for i in trunk if f"trunk.{i}.weight" in state]
        self.mean = (state["mean.weight"].float().numpy().T, state["mean.bias"].float().numpy())
        self.pedal_gain, self.checkpoint = pedal_gain, str(checkpoint)

    def raw(self, obs) -> np.ndarray:
        """The policy action in [-1, 1]^2, as sac_env.step takes it."""
        x = np.asarray(obs, dtype=np.float64)
        for weight, bias in self.layers:
            x = np.maximum(x @ weight + bias, 0.0)
        return np.tanh(x @ self.mean[0] + self.mean[1])

    def controls(self, obs) -> np.ndarray:
        """(steer, applied pedal): the pedal after sac_env.controls' gain and clip."""
        a = self.raw(obs)
        return np.stack([a[..., 0], np.clip(self.pedal_gain * a[..., 1], -1.0, 1.0)], axis=-1)
