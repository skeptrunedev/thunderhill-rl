"""Exercise real PEFT checkpoint reloads with a tiny arithmetic fixture."""
from pathlib import Path
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model

from alpamayo_policy import AlpamayoPolicy


class CheckpointTests(unittest.TestCase):
    def make_policy(self):
        policy = AlpamayoPolicy.__new__(AlpamayoPolicy)
        policy.model = torch.nn.Module()
        policy.model.expert = get_peft_model(
            torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)),
            LoraConfig(r=1, lora_alpha=2, target_modules=['0']),
        )
        policy.trainable = [p for p in policy.model.parameters() if p.requires_grad]
        policy.optimizer = torch.optim.AdamW(policy.trainable)
        policy.diffusion_steps = 10
        policy.noise_level = .7
        policy.converted_config = {}
        policy.buffer_restoration = {}
        return policy

    def test_distinct_saved_adapters_replace_live_weights(self):
        policy = self.make_policy()
        with tempfile.TemporaryDirectory() as root:
            initial, updated = Path(root) / 'initial', Path(root) / 'updated'
            initial_hash = policy.save(initial)
            with torch.no_grad():
                for parameter in policy.trainable:
                    parameter.add_(.125)
            updated_hash = policy.save(updated)
            self.assertNotEqual(initial_hash, updated_hash)
            fresh = self.make_policy()
            self.assertEqual(fresh.reload(updated), updated_hash)
            self.assertEqual(fresh.reload(initial), initial_hash)
            self.assertEqual(fresh.reload(updated), updated_hash)

    def test_training_resume_restores_adam_moments_and_next_step(self):
        policy = self.make_policy()
        def step(p):
            p.optimizer.zero_grad()
            for parameter in p.trainable:
                parameter.grad = torch.full_like(parameter, .25)
            p.optimizer.step()
        step(policy)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'checkpoint'
            digest = policy.save(path)
            fresh = self.make_policy()
            self.assertEqual(fresh.restore_training(path), digest)
            for parameter in fresh.trainable:
                self.assertEqual(fresh.optimizer.state[parameter]['step'].item(), 1)
            step(policy)
            step(fresh)
            self.assertEqual(policy.fingerprint(), fresh.fingerprint())


if __name__ == '__main__':
    unittest.main()
