"""Orchestrator isolation and actual PEFT checkpoint roundtrip regression tests."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from train_full_lap_grpo import collect, collect_waves, save_checkpoint


class FakeEpisode:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.index = len(self.instances)
        self.instances.append(self)
        self.output = kwargs["output"]
        self.output.mkdir(parents=True)
        self.records = []
        self.done = False
        self.observation = {"sim_time": 0}
        self.closed = False
        self.limit = 1 if self.index == 1 else 2

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def prompt(self):
        if self.done:
            raise AssertionError("Finished lane must retain previous prompt")
        return f"lane {self.index} action {len(self.records)}"

    def apply(self, completion, completion_ids, prompt_ids, *, behavior_logprobs):
        if self.done:
            raise AssertionError("Completed lane advanced")
        self.records.append(dict(completion=completion, completion_ids=completion_ids,
                                 prompt_ids=prompt_ids, behavior_logprobs=behavior_logprobs))
        self.observation["sim_time"] += .1
        self.done = len(self.records) == self.limit

    def finish(self):
        return dict(training_eligible=True, reward_components={"total": float(self.index)}, count=len(self.records))


class FakePolicy:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, greedy_indices=()):
        self.calls.append((list(prompts), greedy_indices))
        return [dict(completion="control", prompt_ids=[10+i], completion_ids=[20+i, 1],
                     old_per_token_logps=[-.2, -.3]) for i, _ in enumerate(prompts)]


class FullLapTests(unittest.TestCase):
    def test_collection_excludes_eval_and_finished_lane_samples(self):
        FakeEpisode.instances = []
        with tempfile.TemporaryDirectory() as directory, patch("train_full_lap_grpo.LapEpisode", FakeEpisode):
            policy = FakePolicy()
            episodes, summary, prompt = collect(policy, None, SimpleNamespace(model="test", revision="test"),
                "godot", Path(directory), "0" * 64, 4, 2, 1)
            self.assertEqual([len(e["decisions"]) for e in episodes], [1, 2])
            self.assertEqual([e["reward_components"]["total"] for e in episodes], [1, 2])
            self.assertEqual([len(prompts) for prompts, _ in policy.calls], [3, 3])
            self.assertTrue(all(greedy == (0,) for _, greedy in policy.calls))
            self.assertEqual(policy.calls[0][0][1], policy.calls[1][0][1])
            self.assertEqual(episodes[0]["decisions"][0]["old_per_token_logps"], [-.2, -.3])
            self.assertTrue(all(e.closed for e in FakeEpisode.instances))
            self.assertEqual(summary["evaluation"]["count"], 2)
            self.assertTrue((Path(directory) / "collection.json").exists())

    def test_twelve_rollouts_use_four_bounded_waves_and_unique_metadata(self):
        FakeEpisode.instances = []
        with tempfile.TemporaryDirectory() as directory, patch("train_full_lap_grpo.LapEpisode", FakeEpisode):
            policy = FakePolicy()
            episodes, summary, _ = collect_waves(policy, None, SimpleNamespace(model="test", revision="test"),
                "godot", Path(directory), "0" * 64, 4, 12, 1, 3)
            self.assertEqual(len(episodes), 12)
            self.assertEqual(len(summary["rollouts"]), 12)
            self.assertEqual(len(summary["waves"]), 4)
            self.assertTrue(all(len(prompts) == 4 for prompts, _ in policy.calls))
            sampled = [e for e in FakeEpisode.instances if not e.kwargs["evaluation"]]
            self.assertEqual([e.kwargs["rollout"] for e in sampled], list(range(1, 13)))
            self.assertTrue(all(e.kwargs["rollout_count"] == 12 for e in sampled))
            self.assertEqual(len({e.output for e in FakeEpisode.instances}), 16)
            self.assertEqual([e["reward_components"]["total"] for e in episodes],
                             [float(e.index) for e in sampled])

    def test_heldout_is_repeatable_restores_rng_and_never_returns_training(self):
        class RandomPolicy(FakePolicy):
            def generate(self, prompts, greedy_indices=()):
                self.draws.append(torch.rand(len(prompts)).tolist())
                return super().generate(prompts, greedy_indices)

        torch.manual_seed(73)
        before = torch.random.get_rng_state().clone()
        draws = []
        with tempfile.TemporaryDirectory() as directory, patch("train_full_lap_grpo.LapEpisode", FakeEpisode):
            for index in range(2):
                FakeEpisode.instances = []
                policy = RandomPolicy()
                policy.draws = []
                episodes, result, _ = collect_waves(policy, None, SimpleNamespace(model="test", revision="test"),
                    "godot", Path(directory) / str(index), "0" * 64, index, 5, 1, 3, evaluation_seed=1073)
                self.assertEqual(episodes, [])
                self.assertTrue(result["evaluation_only"])
                self.assertEqual(result["evaluation_seed"], 1073)
                self.assertEqual(len(result["rollouts"]), 5)
                self.assertTrue(all(e.kwargs["evaluation"] for e in FakeEpisode.instances))
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                draws.append(policy.draws)
            self.assertEqual(draws[0], draws[1])

    def test_actual_peft_roundtrip_restores_trainable_parameters_and_optimizer_identity(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        base = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=.01)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3}, unk_token="<unk>")),
            pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory, patch("train_full_lap_grpo.write_spec"):
            result = save_checkpoint(model, tokenizer, None, SimpleNamespace(optimizer=optimizer),
                                     Path(directory) / "checkpoint", "hello")
            self.assertTrue(result["reloaded_logits_match"])
            self.assertEqual(model.active_adapter, "default")
            self.assertNotIn("roundtrip", model.peft_config)
            after = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
            self.assertEqual(set(trainable), set(after))
            self.assertTrue(all(parameter is after[name] for name, parameter in trainable.items()))
            self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]}, {id(p) for p in after.values()})
            model.train()
            model(torch.tensor([[3, 4]])).logits.sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in after.values()))
            self.assertTrue(any(p.grad.abs().sum() > 0 for p in after.values()))
            self.assertTrue((Path(directory) / "checkpoint" / "optimizer.pt").exists())


if __name__ == "__main__":
    unittest.main()
