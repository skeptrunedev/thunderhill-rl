"""CPU regressions for resident embedding inference; no downloads or GPU use."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

from local_inference import FiniteLogits, compare_reference, load_local_policy, place_resident_embedding
from model_runtime import GEMMA4_SPEC, LEGACY_SPEC


class LocalInferenceTests(unittest.TestCase):
    def test_finite_logit_check_preserves_scores_and_rejects_nan_infinity(self):
        check = FiniteLogits()
        inputs = torch.tensor([[2]])
        scores = torch.tensor([[0.0, -8.0, 3.0]], dtype=torch.float16)
        original = scores.clone()
        self.assertIs(check(inputs, scores), scores)
        torch.testing.assert_close(scores, original, rtol=0, atol=0)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(FloatingPointError, "nonfinite"):
                check(inputs, torch.tensor([[0.0, value]], dtype=torch.float16))

    def test_reference_rejects_nonpositive_sample_count(self):
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive"):
                compare_reference(Mock(), Mock(), Path("unused"), "expected", count=count)

    def test_profile_rejects_wrong_identity_before_loading(self):
        with patch("local_inference.load_base") as load:
            with self.assertRaisesRegex(ValueError, "pinned Gemma4"):
                load_local_policy(LEGACY_SPEC, Path("unused"))
            load.assert_not_called()

    def test_profile_requires_cuda_before_loading(self):
        with patch("torch.cuda.is_available", return_value=False), patch(
            "local_inference.load_base"
        ) as load:
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                load_local_policy(GEMMA4_SPEC, Path("unused"))
            load.assert_not_called()

    def test_adapter_loading_cannot_automatically_dispatch_to_gpu(self):
        base, policy = Mock(), Mock()
        policy.eval.return_value = policy
        with patch("torch.cuda.is_available", return_value=True), patch(
            "local_inference.load_base", return_value=base
        ) as load, patch(
            "local_inference.PeftModel.from_pretrained", return_value=policy
        ) as adapter, patch("local_inference.place_resident_embedding") as place:
            self.assertIs(load_local_policy(GEMMA4_SPEC, Path("adapter")), policy)
        load.assert_called_once_with(GEMMA4_SPEC, device="cpu", dtype="float16")
        adapter.assert_called_once_with(
            base, "adapter", torch_device="cpu", device_map={"": "cpu"},
            autocast_adapter_dtype=True,
        )
        place.assert_called_once_with(base, "cuda:0")

    def test_actual_adapter_preserves_weights_logits_and_cached_generation(self):
        torch.manual_seed(17)
        config = Gemma4TextConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, global_head_dim=8,
            hidden_size_per_layer_input=4, vocab_size_per_layer_input=32,
            layer_types=["sliding_attention", "full_attention"],
        )
        base = Gemma4ForCausalLM(config).to(torch.float16)
        initial = {key: value.clone() for key, value in base.state_dict().items()}
        model = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        )).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.1)
        inputs = {"input_ids": torch.tensor([[2, 3, 4]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
        with torch.inference_mode():
            expected = model(**inputs, logits_to_keep=1).logits.clone()
            generated = model.generate(**inputs, max_new_tokens=3, do_sample=False)
            with model.disable_adapter():
                unadapted = model(**inputs, logits_to_keep=1).logits
        self.assertGreater((expected - unadapted).abs().max().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored_base = Gemma4ForCausalLM(config).to(torch.float16)
            restored_base.load_state_dict(initial)
            # Exercise PEFT's real CPU dispatch branch, which otherwise defaults
            # to automatic placement when the base has hf_device_map metadata.
            restored_base.hf_device_map = {"": "cpu"}
            restored = PeftModel.from_pretrained(
                restored_base, directory, torch_device="cpu", device_map={"": "cpu"},
                autocast_adapter_dtype=True,
            ).eval()
            original = {key: value.clone() for key, value in restored.state_dict().items()}
            place_resident_embedding(restored_base, "cpu")
            with torch.inference_mode():
                actual = restored(**inputs, logits_to_keep=1).logits
                actual_generated = restored.generate(**inputs, max_new_tokens=3, do_sample=False)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_generated, generated, rtol=0, atol=0)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        self.assertIs(restored_base.lm_head.weight, restored_base.model.embed_tokens.weight)
        hook = restored_base.model.embed_tokens_per_layer._hf_hook
        self.assertFalse(hook.offload)
        self.assertTrue(hook.io_same_device)
        self.assertEqual(str(hook.execution_device), "cpu")
        self.assertTrue(all(parameter.dtype == torch.float32 for name, parameter in restored.named_parameters() if "lora_" in name))

    def test_reference_rejects_wrong_adapter_and_empty_file_before_inference(self):
        model, tokenizer = Mock(), Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decisions.jsonl"
            for contents in ("", json.dumps({"adapter_sha256": "different"}) + "\n"):
                path.write_text(contents)
                with self.assertRaisesRegex(ValueError, "exact adapter"):
                    compare_reference(model, tokenizer, path, "expected")
        model.assert_not_called()
        tokenizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
