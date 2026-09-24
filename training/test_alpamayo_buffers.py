"""Validate release configuration buffer repair using actual upstream modules."""
import copy
import os
from pathlib import Path
import unittest

import torch

from alpamayo_policy import CONFIG_DERIVED_BUFFERS, configure_sources, restore_release_buffers


class ReleaseBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ.get('ALPAMAYO_SOURCE_ROOT', '/opt/alpamayo-sources'))
        if not root.is_dir():
            raise unittest.SkipTest('Pinned NVIDIA sources required')
        configure_sources(root)
        from hydra.utils import instantiate
        cls.instantiate = staticmethod(instantiate)

    def make_model(self):
        config = {
            'action_in_proj_cfg': {
                '_target_': 'alpagym_alpamayo_r1.submodules.action_in_proj.PerWaypointActionInProjV2',
                'num_fourier_feats': 20, 'max_freq': 100.,
                'hidden_size': 16, 'num_enc_layers': 2,
            },
            'action_space_cfg': {
                '_target_': 'alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace',
                'accel_mean': 0.02902694707164455, 'accel_std': 0.6810426736454882,
                'curvature_mean': 0.0002692167976330542,
                'curvature_std': 0.026148280660833106,
            },
        }
        model = torch.nn.Module()
        model.action_in_proj = self.instantiate(config['action_in_proj_cfg'], in_dims=(64, 2), out_dim=32)
        model.action_space = self.instantiate(config['action_space_cfg'])
        return model.to(dtype=torch.bfloat16), config

    def corrupt(self, model, *, meta=False):
        for name in CONFIG_DERIVED_BUFFERS:
            parent, key = name.rsplit('.', 1)
            module = model.get_submodule(parent)
            old = module.get_buffer(key)
            bad = torch.empty_like(old, device='meta') if meta else torch.full_like(old, float('nan'))
            module.register_buffer(key, bad)

    def test_reconstructs_real_values_and_preserves_learned_weights_and_rng(self):
        model, config = self.make_model()
        expected = {name: model.get_buffer(name).clone() for name in CONFIG_DERIVED_BUFFERS}
        parameters = {name: p.clone() for name, p in model.named_parameters()}
        self.corrupt(model)
        rng = torch.random.get_rng_state().clone()
        report = restore_release_buffers(model, {'missing_keys': list(CONFIG_DERIVED_BUFFERS)}, config)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(set(report['restored']), CONFIG_DERIVED_BUFFERS)
        for name, value in expected.items():
            torch.testing.assert_close(model.get_buffer(name), value, rtol=0, atol=0)
            self.assertFalse(model.get_buffer(name).requires_grad)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, parameters[name], rtol=0, atol=0)
        self.assertEqual(len(report['config_sha256']), 64)

    def test_reconstructs_meta_buffers_on_loaded_parameter_device(self):
        model, config = self.make_model()
        self.corrupt(model, meta=True)
        restore_release_buffers(model, {'missing_keys': list(CONFIG_DERIVED_BUFFERS)}, config)
        for name in CONFIG_DERIVED_BUFFERS:
            self.assertEqual(model.get_buffer(name).device.type, 'cpu')
            self.assertTrue(torch.isfinite(model.get_buffer(name)).all())

    def test_rejects_missing_learned_weight_and_other_load_errors(self):
        for info in (
            {'missing_keys': ['action_in_proj.encoder.trunk.0.weight']},
            {'unexpected_keys': ['wrong.weight']},
            {'mismatched_keys': [('action_space.accel_mean', [2], [])]},
            {'error_msgs': ['checkpoint failure']},
        ):
            model, config = self.make_model()
            before = copy.deepcopy(model.state_dict())
            with self.subTest(info=info), self.assertRaisesRegex(RuntimeError, 'did not load exactly'):
                restore_release_buffers(model, info, config)
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)

    def test_wrong_registered_shape_fails_before_mutation(self):
        model, config = self.make_model()
        model.action_space.register_buffer('accel_std', torch.zeros(2))
        with self.assertRaisesRegex(RuntimeError, 'buffer contract'):
            restore_release_buffers(model, {'missing_keys': list(CONFIG_DERIVED_BUFFERS)}, config)


if __name__ == '__main__':
    unittest.main()
