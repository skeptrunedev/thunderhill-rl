"""CPU identity and prompt compatibility tests; no model weight download."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
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
        tokenizer.convert_tokens_to_ids.return_value = 106
        tokenizer.apply_chat_template.return_value = "formatted"
        features = {"speed": 8, "curves": [0, 0.1]}
        self.assertEqual(
            PolicyRoadTelemetry(GEMMA4_SPEC, tokenizer).prompt_features(features),
            "formatted",
        )
        messages = tokenizer.apply_chat_template.call_args.args[0]
        prompt = messages[0]["content"]
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("complete the lap as quickly as possible while staying on track", prompt)
        self.assertIn("exactly four integers", prompt)
        self.assertIn("Do not use decimal values or negative pedal values", prompt)
        self.assertIn(json.dumps(features, separators=(",", ":")), prompt)
        self.assertIn("control_bike STEER_MILLI THROTTLE_PERCENT FRONT_PERCENT REAR_PERCENT", prompt)
        self.assertEqual(tokenizer.apply_chat_template.call_args.kwargs, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })
        self.assertEqual(tokenizer.eos_token, "<turn|>")

    def test_wrong_turn_terminator_rejected(self):
        tokenizer = Mock()
        tokenizer.convert_tokens_to_ids.return_value = 1
        with self.assertRaisesRegex(ValueError, "turn terminator"):
            PolicyRoadTelemetry(GEMMA4_SPEC, tokenizer)

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

    def test_bf16_accelerate_adapter_reload_preserves_logits(self):
        # Accelerator precision is process global. A fresh interpreter prevents
        # this CPU regression from changing the rest of the test suite's state.
        result = subprocess.run(
            [sys.executable, "-c", "from test_model_runtime import _check_bf16_reload; _check_bf16_reload()"],
            cwd=Path(__file__).resolve().parent,
            text=True, capture_output=True, timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _check_bf16_reload():
    import torch
    from accelerate import Accelerator
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from model_runtime import inference_precision

    torch.manual_seed(42)
    config = Gemma4TextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
        hidden_size_per_layer_input=4, vocab_size_per_layer_input=32,
        layer_types=["sliding_attention", "full_attention"],
    )
    base = Gemma4ForCausalLM(config).to(torch.bfloat16)
    original_weights = {key: value.clone() for key, value in base.state_dict().items()}
    model = get_peft_model(base, LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    # Exercise actual nonzero adaptation, rather than the zero initialized B matrix.
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                parameter.normal_(std=0.01)
    accelerator = Accelerator(cpu=True, mixed_precision="bf16")
    model = accelerator.prepare(model).eval()
    inputs = torch.tensor([[2, 3, 4]])
    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        restored_base = Gemma4ForCausalLM(config).to(torch.bfloat16)
        restored_base.load_state_dict(original_weights)
        reloaded = PeftModel.from_pretrained(restored_base, directory).eval()
        with torch.inference_mode(), inference_precision(model):
            expected = model(input_ids=inputs, logits_to_keep=1).logits
        with torch.inference_mode(), inference_precision(reloaded):
            actual = reloaded(input_ids=inputs, logits_to_keep=1).logits
    assert expected.dtype == torch.float32
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(expected.float(), actual.float(), rtol=1e-5, atol=1e-5)
    assert (expected.float() - actual.float()).abs().max().item() == 0.0


if __name__ == "__main__":
    unittest.main()
