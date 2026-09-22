"""CPU identity and prompt compatibility tests; no model weight download."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from lap_policy import RoadTelemetry
from model_runtime import (
    GEMMA4_SPEC,
    LEGACY_SPEC,
    PolicyRoadTelemetry,
    load_base,
    read_spec,
    write_spec,
)


class ModelRuntimeTests(unittest.TestCase):
    def test_legacy_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            self.assertEqual(read_spec(adapter), LEGACY_SPEC)
            write_spec(adapter, GEMMA4_SPEC)
            self.assertEqual(read_spec(adapter), GEMMA4_SPEC)

    def test_wrong_or_unidentified_adapter_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(
                json.dumps(
                    {
                        "base_model_name_or_path": GEMMA4_SPEC.model,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Adapter base model"):
                read_spec(adapter)
            write_spec(adapter, GEMMA4_SPEC)
            self.assertEqual(read_spec(adapter), GEMMA4_SPEC)
            with self.assertRaises(ValueError):
                write_spec(adapter, LEGACY_SPEC)

    def test_unpinned_revision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                write_spec(Path(directory), replace(GEMMA4_SPEC, revision="main"))

    def test_legacy_prompt_is_byte_identical(self):
        tokenizer = Mock()
        features = {"speed": 8, "curves": [0, 0.1]}
        self.assertEqual(
            PolicyRoadTelemetry(LEGACY_SPEC, tokenizer).prompt_features(features),
            RoadTelemetry().prompt_features(features),
        )
        tokenizer.apply_chat_template.assert_not_called()

    def test_gemma4_uses_non_thinking_chat_with_exact_telemetry(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "formatted"
        features = {"speed": 8, "curves": [0, 0.1]}
        self.assertEqual(
            PolicyRoadTelemetry(GEMMA4_SPEC, tokenizer).prompt_features(features),
            "formatted",
        )
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": RoadTelemetry().prompt_features(features)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def test_loader_uses_text_class_and_pinned_dtype(self):
        import torch

        with patch("transformers.Gemma4ForCausalLM.from_pretrained") as loader:
            from model_runtime import GEMMA4_KEY_MAPPING
            model = Mock()
            loader.return_value = (model, {})
            result = load_base(GEMMA4_SPEC, device="cpu")
            self.assertIs(result, model)
            loader.assert_called_once_with(
                GEMMA4_SPEC.model,
                revision=GEMMA4_SPEC.revision,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map={"": "cpu"},
                output_loading_info=True,
                key_mapping=GEMMA4_KEY_MAPPING,
            )

    def test_loader_rejects_missing_text_and_unexpected_keys(self):
        from model_runtime import _validate_loading_info
        _validate_loading_info(GEMMA4_SPEC, {"unexpected_keys": ["model.vision_tower.a", "model.embed_audio.weight"]})
        for info in (
            {"missing_keys": ["model.embed_tokens_per_layer.weight"]},
            {"unexpected_keys": ["model.language_model.layers.0.mlp.up_proj.weight"]},
            {"mismatched_keys": ["model.embed_tokens.weight"]},
            {"error_msgs": ["invalid"]},
        ):
            with self.assertRaisesRegex(ValueError, "did not load completely"):
                _validate_loading_info(GEMMA4_SPEC, info)

    def test_multimodal_checkpoint_load_preserves_all_text_weights(self):
        import torch
        from safetensors.torch import save_file
        from transformers import Gemma4Config, Gemma4ForCausalLM, Gemma4TextConfig
        from model_runtime import GEMMA4_KEY_MAPPING, _validate_loading_info

        config = Gemma4TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
            hidden_size_per_layer_input=4, vocab_size_per_layer_input=32,
            layer_types=["sliding_attention", "full_attention"],
        )
        source = Gemma4ForCausalLM(config)
        weights = {
            key.replace("model.", "model.language_model.", 1): value.clone()
            for key, value in source.state_dict().items() if key != "lm_head.weight"
        }
        weights["model.embed_audio.weight"] = torch.ones(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            Gemma4Config(text_config=config).save_pretrained(directory)
            save_file(weights, str(Path(directory) / "model.safetensors"))
            loaded, info = Gemma4ForCausalLM.from_pretrained(
                directory, output_loading_info=True, key_mapping=GEMMA4_KEY_MAPPING,
            )
        _validate_loading_info(GEMMA4_SPEC, info)
        for key, expected in source.state_dict().items():
            self.assertTrue(torch.equal(expected, loaded.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
