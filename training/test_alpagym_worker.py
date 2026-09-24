"""Recording identity tests; these do not pretend to execute NVIDIA training."""

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from training.alpagym_worker import publish_identity, register_recorded_rollout


class RecordingIdentityTests(unittest.TestCase):
    def test_unknown_version_is_rejected_before_publishing(self):
        with tempfile.TemporaryDirectory() as root:
            for version in (None, -1, True):
                with self.assertRaises(ValueError):
                    publish_identity(
                        Path(root),
                        1234,
                        policy_version=version,
                        batch_id="batch",
                        is_validation=False,
                        active=True,
                    )
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_wrapper_forwards_unchanged_and_releases_identity_on_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rollout_identity/driver_1234.json"
            marker = object()
            observed = []

            class UpstreamFixture:
                def rollout_generation(self, *args, **kwargs):
                    observed.append((args, kwargs, json.loads(path.read_text())))
                    if kwargs["is_validation"]:
                        raise RuntimeError("fixture failure")
                    return marker

            class RegistryFixture:
                @staticmethod
                def register(name):
                    assert name == "thunderhill_alpagym_rollout"
                    return lambda cls: cls

            upstream = ModuleType("alpagym_runtime.cosmos.rollout_backend")
            upstream.AlpagymRollout = UpstreamFixture
            registry = ModuleType("cosmos_rl.rollout.rollout_base")
            registry.RolloutRegistry = RegistryFixture
            with patch.dict(
                sys.modules, {upstream.__name__: upstream, registry.__name__: registry}
            ):
                adapter = register_recorded_rollout()()
            adapter._run_config = SimpleNamespace(
                artifact_paths=SimpleNamespace(run_dir=Path(root)),
                cosmos=SimpleNamespace(rollout=SimpleNamespace(prefetch_rollout=False)),
            )
            adapter._driver_server = SimpleNamespace(port=1234)
            payloads = [object()]
            result = adapter.rollout_generation(
                payloads, "stream", "packer", current_weight_version=7
            )
            self.assertIs(result, marker)
            self.assertIs(observed[0][0][0], payloads)
            self.assertEqual(observed[0][1]["current_weight_version"], 7)
            self.assertEqual(observed[0][2]["policy_version"], 7)
            self.assertTrue(observed[0][2]["active"])
            self.assertFalse(json.loads(path.read_text())["active"])
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                adapter.rollout_generation(
                    payloads,
                    "stream",
                    "packer",
                    is_validation=True,
                    current_weight_version=8,
                )
            self.assertFalse(json.loads(path.read_text())["active"])
            adapter._run_config.cosmos.rollout.prefetch_rollout = True
            with self.assertRaisesRegex(ValueError, "prefetch"):
                adapter.rollout_generation(
                    payloads, "stream", "packer", current_weight_version=9
                )


if __name__ == "__main__":
    unittest.main()
