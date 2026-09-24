"""CPU tests of the official stochastic action distribution and our update.

These use a tiny random action head to test arithmetic, not driving capability.
No model is trained from demonstrations, and no fixture is a driving checkpoint.
"""
import os
from pathlib import Path
import unittest

import torch

from alpamayo_policy import AlpamayoPolicy, configure_sources


class OfficialFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get("ALPAMAYO_SOURCE_ROOT", "/opt/alpamayo-sources"))
        if not root.is_dir():
            raise unittest.SkipTest("Pinned NVIDIA source checkout required")
        configure_sources(root)
        from alpamayo1_x_rl.models.expert_model.model import ExpertModelRL
        from alpamayo1_x_rl.diffusion.flow_matching import FlowMatching
        cls.expert_class = ExpertModelRL
        cls.flow_class = FlowMatching

    def test_real_processor_camera_and_history_contract(self):
        import importlib.util
        import json
        from types import SimpleNamespace
        import numpy as np
        processor_path = os.environ.get("ALPAMAYO_PROCESSOR_PATH")
        config_path = os.environ.get("ALPAMAYO_CONFIG_PATH")
        if not processor_path or not config_path:
            self.skipTest("Gated pinned processor and release config paths required")
        root = Path(os.environ["ALPAMAYO_SOURCE_ROOT"])
        source = root / "alpagym/packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py"
        spec = importlib.util.spec_from_file_location("test_alpa_conversion", source)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        from alpamayo1_x_rl.models.expert_model.config import ExpertModelConfig
        config = ExpertModelConfig(**converter.build_expert_config(
            json.loads(Path(config_path).read_text()), processor_path,
        ))
        policy = AlpamayoPolicy.__new__(AlpamayoPolicy)
        policy.device = "cpu"
        policy.model = SimpleNamespace(config=config, processor=config._build_processor())
        frames = np.zeros((4, 360, 640, 3), dtype=np.uint8)
        rotation = np.repeat(np.eye(3)[None], 16, axis=0)
        prepared = policy._prepare(frames, np.zeros((16, 3)), rotation, "Race Thunderhill East.")
        self.assertEqual(prepared["ego_history_xyz"].shape, (1, 1, 16, 3))
        self.assertEqual(prepared["ego_history_rot"].shape, (1, 1, 16, 3, 3))
        tokens = prepared["tokenized_data"]
        self.assertEqual(tokens["image_grid_thw"].shape[0], 4)
        text = policy.model.processor.tokenizer.decode(tokens["input_ids"][0])
        self.assertIn("Front camera", text)
        self.assertIn("frame 3", text)
        self.assertIn("Race Thunderhill East.", text)
        self.assertNotIn("labels", tokens)

    def make_policy(self):
        flow = self.flow_class(x_dims=(8, 2), num_inference_steps=4)

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.coefficient = torch.nn.Parameter(torch.tensor(0.1))

            def forward(self, x, t):
                return self.coefficient * torch.tanh(x)

            def cfm_logprob_sde(self, data, samples, timesteps, noise_level, teacher):
                predicted = self(samples[:, :-1], timesteps[:, :-1])
                return flow._batched_sde_logprob(
                    predicted, timesteps, samples, noise_level=noise_level,
                )

        policy = AlpamayoPolicy.__new__(AlpamayoPolicy)
        policy.device = "cpu"
        policy.model = Head()
        policy.trainable = list(policy.model.parameters())
        policy.optimizer = torch.optim.AdamW(policy.trainable, lr=1e-3, weight_decay=0)
        return policy, flow

    def sample(self, policy, flow, seed):
        result = flow.sample(
            batch_size=1, step_fn=policy.model, device=torch.device("cpu"),
            int_method="sde", return_info=True, return_all_steps=True,
            noise_level=.7, generator=torch.Generator().manual_seed(seed),
        )
        return {"data": {}, "samples_list": result["all_steps"],
                "timesteps": result["timesteps"][None], "noise_level": .7,
                "old_logprob": result["log_prob"].reshape(())}

    def test_sampled_density_replays_with_gradient(self):
        policy, flow = self.make_policy()
        replay = self.sample(policy, flow, 42)
        new = policy.logprob(replay)
        torch.testing.assert_close(new, replay["old_logprob"], atol=1e-6, rtol=1e-6)
        new.backward()
        self.assertTrue(torch.isfinite(policy.model.coefficient.grad))
        self.assertGreater(abs(policy.model.coefficient.grad.item()), 0)

    def test_signed_episode_rewards_change_policy_without_labels(self):
        policy, flow = self.make_policy()
        episodes = [{"reward": 2., "replays": [self.sample(policy, flow, 4)]},
                    {"reward": -1., "replays": [self.sample(policy, flow, 5)]}]
        result = policy.update(episodes, learning_rate=1e-3, max_decisions=24)
        self.assertEqual(result["optimizer_updates"], 1)
        self.assertEqual(result["advantages"], [1., -1.])
        self.assertEqual(result["max_decisions"], 24)
        self.assertNotEqual(result["adapter_sha256_before"], result["adapter_sha256_after"])

    def test_equal_rewards_and_corrupt_density_rejected(self):
        policy, flow = self.make_policy()
        episodes = [{"reward": 1., "replays": [self.sample(policy, flow, seed)]}
                    for seed in (1, 2)]
        with self.assertRaisesRegex(RuntimeError, "Equal gameplay rewards"):
            policy.update(episodes)
        episodes[1]["reward"] = -1.
        episodes[0]["replays"][0]["old_logprob"] += 1.
        with self.assertRaisesRegex(RuntimeError, "density replay mismatch"):
            policy.update(episodes)

    def test_start_conditions_do_not_compete_for_advantage(self):
        policy, flow = self.make_policy()
        episodes = [dict(reward=reward, reward_group=group, replays=[self.sample(policy, flow, seed)])
                    for seed, (group, reward) in enumerate(
                        [('standing', 1.), ('standing', 3.), ('moving', 101.), ('moving', 103.)])]
        result = policy.update(episodes)
        self.assertEqual(result['advantages'], [-1., 1., -1., 1.])
        self.assertEqual(result['reward_group_statistics']['moving']['mean'], 102.)

    def test_start_offset_alone_cannot_produce_learning(self):
        policy, flow = self.make_policy()
        episodes = [dict(reward=reward, reward_group=group, replays=[self.sample(policy, flow, seed)])
                    for seed, (group, reward) in enumerate(
                        [('standing', 1.), ('standing', 1.), ('moving', 100.), ('moving', 100.)])]
        with self.assertRaisesRegex(RuntimeError, 'Equal gameplay rewards'):
            policy.update(episodes)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            policy.update(episodes[:3])


if __name__ == "__main__":
    unittest.main()
