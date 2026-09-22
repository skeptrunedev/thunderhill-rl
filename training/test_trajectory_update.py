"""Actual CPU model tests for full trajectory loss and bounded accumulation."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from trajectory_update import TrajectoryConfig, TrajectoryUpdater


class NativeTrajectoryTests(unittest.TestCase):
    def test_actual_native_generation_matches_trl_masked_likelihood_and_updates(self):
        from batched_policy import BatchedPolicy
        from model_runtime import GEMMA4_SPEC
        from native_tools import NativeBikeTools
        from native_constraints import NativeToolConstraint
        torch.set_num_threads(1)
        torch.manual_seed(8)
        tokenizer = AutoTokenizer.from_pretrained(
            GEMMA4_SPEC.model, revision=GEMMA4_SPEC.revision, local_files_only=True)
        tokenizer.eos_token = "<|tool_response>"
        native = NativeBikeTools(tokenizer)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
            max_position_embeddings=2048, attention_dropout=0.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=50))
        constraint = NativeToolConstraint(tokenizer, len(tokenizer))
        policy = BatchedPolicy(model, tokenizer, native_tools=native, constraints=constraint)
        decisions = policy.generate([native.prompt({"speed": 5}), native.prompt({"speed": 10})])
        episodes = [dict(reward=reward, decisions=[decision])
                    for reward, decision in zip((1.0, -1.0), decisions, strict=True)]
        with tempfile.TemporaryDirectory() as directory:
            updater = TrajectoryUpdater(model, tokenizer, directory,
                TrajectoryConfig(max_actions=2, microbatch_size=2, max_completion_length=128),
                constraints=constraint)
            result = updater.update(episodes, generation=1)
        self.assertTrue(result["native_grammar_likelihoods"])
        self.assertLess(result["max_behavior_logp_difference"], 1e-5)
        self.assertGreater(result["parameter_delta_l1"], 0)
        self.assertEqual(result["eos_tokens"], 2)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i for i in range(3, 32)}}
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")), pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        self.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, attention_dropout=0.0, pad_token_id=0, eos_token_id=1))

    def updater(self, model=None, batch=1):
        return TrajectoryUpdater(model or self.model, self.tokenizer, self.directory.name, TrajectoryConfig(max_actions=8, microbatch_size=batch))

    def decision(self, prompt, completion):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor([prompt + completion])).logits[0]
            scores = logits[len(prompt)-1:-1].float().log_softmax(-1)
            old = scores.gather(1, torch.tensor(completion)[:, None]).flatten().tolist()
        return dict(prompt_ids=prompt, completion_ids=completion, old_per_token_logps=old)

    def episodes(self):
        return [
            dict(reward=1.0, decisions=[self.decision([3, 4], [5, 1]), self.decision([6, 7, 8], [9, 10, 1])]),
            dict(reward=-1.0, decisions=[self.decision([3, 4], [11, 1])]),
        ]

    def test_chunk_invariance_and_behavior_likelihood(self):
        episodes = self.episodes()
        model2 = copy.deepcopy(self.model)
        first = self.updater(batch=1)
        second = self.updater(model2, batch=3)
        audit1 = first.accumulate(episodes)
        audit2 = second.accumulate(episodes)
        self.assertAlmostEqual(audit1["loss"], audit2["loss"], places=7)
        self.assertLess(audit2["max_behavior_logp_difference"], 1e-5)
        for left, right in zip(first.parameters, second.parameters, strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-7)
        self.assertEqual(audit1["trained_tokens"], 7)
        self.assertEqual(audit1["later_actions"], 1)
        self.assertEqual(audit1["eos_tokens"], 3)
        self.assertEqual(audit1["fixed_normalizer"], 2 * 8 * 32)

    def test_later_action_changes_gradient_and_real_update(self):
        episodes = self.episodes()
        updater = self.updater()
        updater.accumulate(episodes)
        full = [p.grad.clone() for p in updater.parameters]
        truncated = copy.deepcopy(episodes)
        truncated[0]["decisions"] = truncated[0]["decisions"][:1]
        updater.accumulate(truncated)
        self.assertTrue(any(not torch.allclose(g, p.grad) for g, p in zip(full, updater.parameters)))
        result = updater.update(episodes, generation=1)
        self.assertGreater(result["parameter_delta_l1"], 0)
        self.assertEqual(result["optimizer_steps"], 1)

    def test_padding_is_masked_and_cannot_change_loss_or_gradient(self):
        updater = self.updater(batch=3)
        rows, _ = updater.rows(self.episodes())
        batch = updater.batch(rows)
        self.model.train()
        loss = updater.trainer.compute_loss(self.model, batch)
        loss.backward()
        grads = [p.grad.clone() for p in updater.parameters]
        self.model.zero_grad()
        changed = {k: v.clone() for k, v in batch.items()}
        changed["completion_ids"][changed["completion_mask"] == 0] = 19
        changed["old_per_token_logps"][changed["completion_mask"] == 0] = -20
        loss2 = updater.trainer.compute_loss(self.model, changed)
        loss2.backward()
        torch.testing.assert_close(loss, loss2)
        for old, parameter in zip(grads, updater.parameters):
            torch.testing.assert_close(old, parameter.grad)

    def test_lora_update_preserves_precision_and_frozen_base(self):
        from peft import LoraConfig, get_peft_model

        self.model = get_peft_model(self.model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        frozen = {n: p.detach().clone() for n, p in self.model.named_parameters() if not p.requires_grad}
        updater = self.updater(batch=2)
        audit = updater.update(self.episodes(), generation=2)
        self.assertGreater(audit["parameter_delta_l1"], 0)
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.assertEqual(parameter.dtype, torch.float32)
            else:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        with self.assertRaises(FileExistsError):
            updater.update(self.episodes(), generation=2)

    def test_profiler_preserves_weights_mode_and_selects_measured_throughput(self):
        episodes = self.episodes()
        updater = self.updater()
        before = [p.detach().clone() for p in self.model.parameters()]
        self.model.eval()
        report = updater.profile_microbatches(episodes, candidates=(1, 2, 4))
        self.assertFalse(self.model.training)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(updater.optimizer.state)
        self.assertEqual(report["selected_microbatch_size"], max(report["probes"], key=lambda p: p["actions_per_second"])["microbatch_size"])
        self.assertEqual(updater.config.microbatch_size, report["selected_microbatch_size"])
        self.assertEqual(report["optimizer_state_reserve_bytes"], 2 * sum(p.numel() * p.element_size() for p in updater.parameters))
        for old, parameter in zip(before, self.model.parameters()):
            torch.testing.assert_close(old, parameter, atol=0, rtol=0)
            self.assertIsNone(parameter.grad)

    def test_profile_failure_restores_mode_and_loss_audit_needs_one_forward(self):
        episodes = self.episodes()
        updater = self.updater(batch=3)
        original = updater.trainer._get_per_token_logps_and_entropies
        with patch.object(updater.trainer, "_get_per_token_logps_and_entropies", wraps=original) as forward:
            audit = updater.accumulate(episodes)
            self.assertEqual(forward.call_count, 1)
            self.assertLess(audit["max_behavior_logp_difference"], 1e-5)
        self.model.eval()
        with patch.object(updater.trainer, "trajectory_loss", side_effect=torch.cuda.OutOfMemoryError("test")):
            with self.assertRaisesRegex(RuntimeError, "No training microbatch"):
                updater.profile_microbatches(episodes)
        self.assertFalse(self.model.training)
        self.assertTrue(all(p.grad is None for p in updater.parameters))

    def test_reject_missing_behavior_tokens_and_after_eos(self):
        updater = self.updater()
        episodes = self.episodes()
        episodes[0]["decisions"][0]["old_per_token_logps"].pop()
        with self.assertRaisesRegex(ValueError, "Every generated token"):
            updater.rows(episodes)
        episodes = self.episodes()
        episodes[0]["decisions"][0]["completion_ids"] = [1, 5]
        with self.assertRaisesRegex(ValueError, "after EOS"):
            updater.rows(episodes)


if __name__ == "__main__":
    unittest.main()
