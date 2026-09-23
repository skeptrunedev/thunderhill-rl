"""CPU architecture qualification, never a source of gameplay training data.

Only the real tokenizer is downloaded. Random tiny hybrid Qwen weights exercise
checkpoint loading and on policy reward gradients without allocating 27B weights.
"""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from batched_policy import BatchedPolicy
from model_runtime import (QWEN27B_SPEC, QWEN_KEY_MAPPING,
                           QWEN_LORA_TARGET_MODULES, _validate_loading_info)
from native_constraints import NativeToolConstraint
from qwen_tools import QwenBikeTools
from trajectory_update import TrajectoryConfig, TrajectoryUpdater


def tiny_config(vocab_size=64, pad_token_id=0, eos_token_id=1):
    return Qwen3_5TextConfig(
        vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=4096, attention_dropout=0.0,
        pad_token_id=pad_token_id, eos_token_id=eos_token_id,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0},
    )


class QwenTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def test_multimodal_namespace_loads_every_text_weight_exactly(self):
        model = Qwen3_5ForCausalLM(tiny_config())
        # The published multimodal checkpoint nests the language backbone.
        state = {name.replace("model.", "model.language_model.", 1)
                 if name.startswith("model.") else name: tensor.clone()
                 for name, tensor in model.state_dict().items()}
        state["model.visual.test_weight"] = torch.ones(1)
        with tempfile.TemporaryDirectory() as directory:
            model.config.save_pretrained(directory)
            save_file(state, str(Path(directory) / "model.safetensors"))
            loaded, info = Qwen3_5ForCausalLM.from_pretrained(
                directory, config=model.config, key_mapping=QWEN_KEY_MAPPING,
                output_loading_info=True, dtype=torch.float32,
            )
        _validate_loading_info(QWEN27B_SPEC, info)
        self.assertFalse(info["missing_keys"])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            _validate_loading_info(QWEN27B_SPEC, {"missing_keys": ["model.embed_tokens.weight"]})

    def test_native_sampled_policy_gradient_updates_and_reloads_lora(self):
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN27B_SPEC.model, revision=QWEN27B_SPEC.revision,
            local_files_only=True,
        )
        native = QwenBikeTools(tokenizer)
        tokenizer.eos_token = native.tool_stop
        base = Qwen3_5ForCausalLM(tiny_config(
            len(tokenizer), tokenizer.pad_token_id, native.stop_token_id))
        original_base = copy.deepcopy(base)
        model = get_peft_model(base, LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=list(QWEN_LORA_TARGET_MODULES), task_type="CAUSAL_LM"))
        frozen = {name: p.detach().clone() for name, p in model.named_parameters()
                  if not p.requires_grad}
        self.assertTrue(any("linear_attn.in_proj_qkv" in name and p.requires_grad
                            for name, p in model.named_parameters()))
        constraint = NativeToolConstraint(tokenizer, len(tokenizer), native_tools=native)
        temperature = 0.6
        policy = BatchedPolicy(model, tokenizer, native_tools=native,
                               constraints=constraint, temperature=temperature)
        # Both actions are sampled from this exact policy. Opposing synthetic
        # rewards test the loss plumbing only; no target actions or SFT labels.
        decisions = policy.generate([native.prompt({"speed": 5}),
                                     native.prompt({"speed": 10})])
        for decision in decisions:
            native.parse_completion(decision["completion"])
            self.assertEqual(decision["completion_ids"][-1], native.stop_token_id)
        episodes = [dict(reward=reward, decisions=[decision])
                    for reward, decision in zip((1.0, -1.0), decisions, strict=True)]
        with tempfile.TemporaryDirectory() as directory:
            updater = TrajectoryUpdater(model, tokenizer, directory,
                TrajectoryConfig(max_actions=1, microbatch_size=1,
                                 max_completion_length=native.max_completion_length,
                                 temperature=temperature, gradient_checkpointing=True),
                constraints=constraint)
            self.assertTrue(model.is_gradient_checkpointing)
            result = updater.update(episodes, generation=1)
            self.assertTrue(result["native_grammar_likelihoods"])
            self.assertLess(result["max_behavior_logp_difference"], 2e-5)
            self.assertGreater(result["parameter_delta_l1"], 0)
            self.assertEqual(result["optimizer_steps"], 1)
            self.assertEqual(result["prompt_tokens_in_loss"], 0)
            self.assertEqual(result["eos_tokens"], 2)
            self.assertEqual(result["advantages"], [1.0, -1.0])
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
            adapter = Path(directory) / "adapter"
            model.save_pretrained(adapter)
            reloaded = PeftModel.from_pretrained(original_base, adapter)
            model.eval()
            reloaded.eval()
            inputs = torch.tensor([decisions[0]["prompt_ids"]])
            with torch.no_grad():
                actual = model(inputs, use_cache=False, logits_to_keep=1).logits
                restored = reloaded(inputs, use_cache=False, logits_to_keep=1).logits
            torch.testing.assert_close(actual, restored, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
