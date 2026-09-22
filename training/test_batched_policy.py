"""CPU verification using an actual tiny random transformer, without downloads."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from batched_policy import BatchedPolicy, GreedyRows, cuda_graph_evidence


class BatchedPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        vocabulary = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "ride": 3, "road": 4,
                      "curve": 5, "control_bike": 6, "0": 7, "20": 8, "left": 9}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=64,
            n_embd=16, n_layer=1, n_head=2, bos_token_id=None, eos_token_id=2,
            pad_token_id=0, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()
        self.policy = BatchedPolicy(self.model, self.tokenizer)

    def test_padding_and_sampled_likelihoods_match_actual_forward(self):
        prompts = ["ride", "ride road curve"]
        rows = self.policy.generate(prompts)
        self.assertEqual([r["prompt_ids"] for r in rows], [[3], [3, 4, 5]])
        for row in rows:
            ids = row["completion_ids"]
            self.assertGreater(len(ids), 0)
            self.assertLessEqual(len(ids), 32)
            self.assertEqual(len(ids), len(row["old_per_token_logps"]))
            if 2 in ids:
                self.assertEqual(ids[-1], 2)
                self.assertEqual(ids.count(2), 1)
            sequence = torch.tensor([row["prompt_ids"] + ids])
            with torch.no_grad():
                logits = self.model(sequence).logits
            start = len(row["prompt_ids"]) - 1
            expected = logits[0, start:start + len(ids)].log_softmax(-1).gather(1, torch.tensor(ids)[:, None])[:, 0]
            torch.testing.assert_close(torch.tensor(row["old_per_token_logps"]), expected, atol=1e-5, rtol=1e-5)

    def test_greedy_row_stops_at_eos_and_other_row_still_sampled(self):
        with torch.no_grad():
            eos = self.model(torch.tensor([[3]])).logits[0, -1].argmax().item()
        token = self.tokenizer.convert_ids_to_tokens(eos)
        self.tokenizer.eos_token = token
        rows = self.policy.generate(["ride", "ride road"], greedy_indices=(0,))
        self.assertEqual(rows[0]["completion_ids"], [eos])
        self.assertEqual(rows[0]["old_per_token_logps"], [0.0])
        self.assertTrue(all(v < 0 for v in rows[1]["old_per_token_logps"]))

    def test_exploration_temperature_is_in_behavior_likelihood(self):
        policy = BatchedPolicy(self.model, self.tokenizer, temperature=3.0)
        row = policy.generate(["ride"])[0]
        ids = row["completion_ids"]
        with torch.no_grad():
            logits = self.model(torch.tensor([row["prompt_ids"] + ids])).logits[0, :len(ids)]
        expected = (logits / 3.0).log_softmax(-1).gather(1, torch.tensor(ids)[:, None])[:, 0]
        torch.testing.assert_close(torch.tensor(row["old_per_token_logps"]), expected, atol=1e-5, rtol=1e-5)

    def test_invalid_temperature_rejected(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "temperature"):
                BatchedPolicy(self.model, self.tokenizer, temperature=value)

    def test_processor_leaves_sampled_row_unchanged_and_rejects_nonfinite(self):
        scores = torch.tensor([[1., 3., 2.], [5., 1., 2.]])
        actual = GreedyRows((0,))(torch.zeros(2, 1, dtype=torch.long), scores)
        torch.testing.assert_close(actual[1], scores[1])
        self.assertEqual(actual[0].argmax().item(), 1)
        self.assertEqual(actual[0, 1].item(), 0)
        with self.assertRaises(FloatingPointError):
            GreedyRows()(None, torch.tensor([[float("nan")]]))

    def test_maximum_completion_is_32_without_eos(self):
        row = self.policy.generate(["ride"], greedy_indices=(0,))[0]
        self.assertEqual(len(row["completion_ids"]), 32)
        self.assertNotIn(self.tokenizer.eos_token_id, row["completion_ids"])

    def test_requested_compilation_cannot_silently_fall_back(self):
        self.policy.compile_inference = True
        self.policy._qualify_compilation(1)
        self.assertFalse(self.policy.compile_qualified)
        with self.assertRaisesRegex(RuntimeError, "was skipped"):
            self.policy._qualify_compilation(2)
        self.model._compiled_call = lambda: None
        self.policy._qualify_compilation(2)
        self.assertTrue(self.policy.compile_qualified)

    def test_default_width_and_overflow_fail_closed(self):
        self.assertEqual(self.policy.compiled_prompt_length, 256)
        self.policy.compile_inference = True
        self.policy.compiled_prompt_length = 1
        with self.assertRaisesRegex(ValueError, "never truncated"):
            self.policy.generate(["ride road"])

    def test_graph_evidence_counts_nodes_not_allocator_graph(self):
        manager = SimpleNamespace(roots={}, graph=object(), path_state=SimpleNamespace(name="NONE"))
        with patch("torch._inductor.cudagraph_trees.get_manager", return_value=manager) as getter:
            evidence = cuda_graph_evidence(torch.device("cuda:0"))
            self.assertFalse(evidence["cuda_graph_captured"])
            getter.assert_called_once_with(0, create_if_none_exists=False)
        child = SimpleNamespace(graph=object(), children={})
        node = SimpleNamespace(graph=object(), children={1: [child]})
        manager.roots = {0: [node]}
        manager.path_state.name = "EXECUTION"
        with patch("torch._inductor.cudagraph_trees.get_manager", return_value=manager):
            evidence = cuda_graph_evidence(torch.device("cuda:0"))
        self.assertEqual(evidence["captured_nodes"], 2)
        self.assertEqual(evidence["path_state"], "EXECUTION")
        self.assertTrue(evidence["cuda_graph_captured"])

    def test_training_mode_restored_and_bad_indices_rejected(self):
        self.model.train()
        self.policy.generate(["ride"], greedy_indices=(0,))
        self.assertTrue(self.model.training)
        for indices in [(1,), (0, 0), (-1,), (False,)]:
            with self.assertRaises(ValueError):
                self.policy.generate(["ride"], greedy_indices=indices)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            self.policy.profile(["ride"])


class NativeBatchedPolicyTests(unittest.TestCase):
    def test_actual_transformer_native_batch_and_fresh_matcher(self):
        from model_runtime import GEMMA4_SPEC
        from native_tools import NativeBikeTools
        tokenizer = AutoTokenizer.from_pretrained(
            GEMMA4_SPEC.model, revision=GEMMA4_SPEC.revision, local_files_only=True,
        )
        native = NativeBikeTools(tokenizer)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_positions=2048, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=None, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
        )).eval()
        policy = BatchedPolicy(model, tokenizer, native_tools=native)
        self.assertEqual(policy.compiled_prompt_length, 1024)
        self.assertEqual(policy.stop_token_id, 50)
        self.assertEqual(policy.max_completion_length, 128)
        # Random tiny weights exercise the constraint, including arbitrary
        # subword splits. This is generation, not a mocked output string.
        prompts = [native.prompt({"speed": 0}), native.prompt({"speed": 10, "lateral": 0})]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            rows = policy.generate(prompts, greedy_indices=(0,))
            second = policy.generate(prompts[:1])
        finally:
            torch.set_num_threads(previous_threads)
        for row in rows + second:
            native.parse_completion(row["completion"])
            self.assertEqual(row["completion_ids"][0], 48)
            self.assertEqual(row["completion_ids"][-2:], [49, 50])
            self.assertLessEqual(len(row["completion_ids"]), 128)
            self.assertTrue(all(torch.isfinite(torch.tensor(row["old_per_token_logps"]))))
            self.assertEqual(len(row["completion_ids"]), len(row["old_per_token_logps"]))
        self.assertTrue(all(value == 0 for value in rows[0]["old_per_token_logps"]))
        self.assertTrue(any(value < 0 for value in rows[1]["old_per_token_logps"]))
        self.assertEqual(rows[0]["prompt_ids"], tokenizer.encode(prompts[0], add_special_tokens=False))


if __name__ == "__main__":
    unittest.main()
