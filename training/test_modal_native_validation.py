"""Migration provenance checks before any native tool campaign proceeds."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from modal_native_validation import validate_campaign, validate_migration, validate_source
from model_runtime import GEMMA4_NATIVE_SPEC, GEMMA4_SPEC, LEGACY_SPEC, write_spec


class MigrationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        write_spec(self.source, GEMMA4_SPEC)
        (self.source / "adapter_model.safetensors").write_bytes(b"source test tensor bytes")
        self.source_hash = validate_source(self.source)
        self.output = self.root / "migration"
        write_spec(self.output / "adapter", GEMMA4_NATIVE_SPEC)
        weights = self.output / "adapter" / "adapter_model.safetensors"
        weights.write_bytes(b"new test tensor bytes")
        self.summary = {
            "completion_masks_verified": True, "reloaded_logits_match": True,
            "supervision_controls_preserved": True,
            "initial_adapter_sha256": self.source_hash,
            "adapter_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        }
        self.save_summary()

    def save_summary(self):
        (self.output / "summary.json").write_text(json.dumps(self.summary))

    def test_native_smoke_requires_protocol_and_valid_sampled_actions(self):
        collection = {"evaluation": {"reason": "episode_tick_limit"},
                      "rollouts": [{"reason": "episode_tick_limit"}]}
        campaign = {"complete": True, "smoke_only": True,
                    "prompt_style": GEMMA4_NATIVE_SPEC.prompt_style,
                    "native_stop_token_id": 50, "constrained_sampling_and_training": True,
                    "generations": [{"collection": collection}],
                    "final_evaluation": {"evaluation": {"reason": "episode_tick_limit"}, "rollouts": []}}
        validate_campaign(campaign)
        for key, value in (("prompt_style", "gemma4_chat"), ("native_stop_token_id", 106),
                           ("constrained_sampling_and_training", False), ("generations", [])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_campaign({**campaign, key: value})
        malformed = copy.deepcopy(campaign)
        malformed["generations"][0]["collection"]["rollouts"][0]["reason"] = "invalid_model_action"
        with self.assertRaisesRegex(ValueError, "invalid model action"):
            validate_campaign(malformed)

    def test_verified_migration_accepted(self):
        self.assertEqual(validate_migration(self.output, self.source_hash), self.summary)

    def test_wrong_model_source_rejected(self):
        other = self.root / "legacy"
        write_spec(other, LEGACY_SPEC)
        with self.assertRaisesRegex(ValueError, "pinned Gemma4"):
            validate_source(other)

    def test_missing_evidence_rejected(self):
        for key in ("completion_masks_verified", "reloaded_logits_match", "supervision_controls_preserved"):
            with self.subTest(key=key):
                self.summary[key] = False
                self.save_summary()
                with self.assertRaisesRegex(ValueError, key):
                    validate_migration(self.output, self.source_hash)
                self.summary[key] = True

    def test_source_lineage_and_output_tampering_rejected(self):
        with self.assertRaisesRegex(ValueError, "source adapter checksum"):
            validate_migration(self.output, "other hash")
        (self.output / "adapter" / "adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "output adapter checksum"):
            validate_migration(self.output, self.source_hash)


if __name__ == "__main__":
    unittest.main()
