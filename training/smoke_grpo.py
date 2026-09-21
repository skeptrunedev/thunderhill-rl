"""Small, real Gemma GRPO update from Godot outcomes, not a racing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_parallel import worker

MODEL = "unsloth/gemma-3-270m-it"
REVISION = "23cf460f6bb16954176b3ddcc8d4f250501458a9"
# A deliberately neutral prompt avoids solving the launch task through prior
# knowledge before the reward path has any variance to validate.
PROMPT = [
    {
        "role": "user",
        "content": "Pick a random letter: A or B. Reply with one letter only.",
    }
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), (
        "CUDA is required; CPU execution cannot pass this test"
    )
    set_seed(71)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
    ).to("cuda")
    prompt = tokenizer.apply_chat_template(
        PROMPT, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    samples = []
    with ExitStack() as stack:
        client, data = stack.enter_context(
            worker(
                args.godot,
                args.output / "environment",
                90,
                ("--agent-max-episode-ticks=120",),
            )
        )
        trace = stack.enter_context((args.output / "rollouts.jsonl").open("w"))

        def game_reward(completions, completion_ids, **kwargs):
            rewards = []
            for completion, ids in zip(completions, completion_ids, strict=True):
                text = (
                    completion[0]["content"]
                    if isinstance(completion, list)
                    else completion
                )
                match = re.fullmatch(r"\s*([AB])\s*", text)
                policy = f"smoke-step-{trainer.state.global_step}"
                obs = client.request({"op": "reset", "policy_id": policy})
                episode = obs["episode_id"]
                reward = -0.1
                transitions = []
                if match:
                    throttle = 0.6 if match[1] == "B" else 0.0
                    while not (obs["terminated"] or obs["truncated"]):
                        obs = client.request(
                            {
                                "op": "advance",
                                "episode_id": episode,
                                "expected_tick": obs["tick"],
                                "action_id": str(obs["tick"]),
                                "controls": {"throttle": throttle},
                            }
                        )
                        if "error" in obs or not obs["rollout_valid"]:
                            raise RuntimeError(f"Invalid simulator rollout: {obs}")
                        transitions.extend(obs["transitions"])
                    reward = sum(
                        t["reward_components"]["legal_progress_m"] for t in transitions
                    )
                record = {
                    "policy_id": policy,
                    "episode_id": episode,
                    "text": text,
                    "completion_ids": ids,
                    "valid_action": bool(match),
                    "reward": reward,
                    "ticks": obs["tick"],
                    "terminated": obs["terminated"],
                    "truncated": obs["truncated"],
                    "reward_version": "launch-probe-v1",
                }
                trace.write(json.dumps(record) + "\n")
                trace.flush()
                samples.append(record)
                rewards.append(reward)
            return rewards

        config = GRPOConfig(
            output_dir=str(args.output / "trainer"),
            max_steps=3,
            per_device_train_batch_size=8,
            num_generations=8,
            gradient_accumulation_steps=1,
            max_completion_length=1,
            learning_rate=1e-4,
            beta=0.0,
            temperature=1.5,
            top_p=1.0,
            top_k=0,
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
            use_vllm=False,
            report_to="none",
            logging_steps=1,
            save_strategy="no",
            mask_truncated_completions=False,
            seed=71,
        )
        trainer = GRPOTrainer(
            model=model,
            args=config,
            reward_funcs=game_reward,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list([{"prompt": PROMPT}] * 24),
            peft_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0,
                task_type="CAUSAL_LM",
            ),
        )
        before = {
            n: p.detach().cpu().clone()
            for n, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        assert before and all("lora_" in n for n in before)
        trainer.model.eval()
        with torch.no_grad():
            before_logits = trainer.model(**inputs).logits[:, -1].detach().cpu()
        trained = trainer.train()
        assert trainer.state.global_step == 3
        delta = max(
            (p.detach().cpu() - before[n]).abs().max().item()
            for n, p in trainer.model.named_parameters()
            if n in before
        )
        assert delta > 0 and torch.isfinite(torch.tensor(delta)), (
            "No finite adapter update"
        )
        assert len({s["reward"] for s in samples if s["valid_action"]}) > 1, (
            "No action dependent game reward contrast"
        )
        trainer.model.eval()
        with torch.no_grad():
            after_logits = trainer.model(**inputs).logits[:, -1].detach().cpu()
        assert torch.isfinite(after_logits).all()
        logit_delta = (after_logits - before_logits).abs().max().item()
        assert logit_delta > 0, "Updated adapters did not affect policy logits"
        checkpoint = args.output / "adapter"
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(str(checkpoint))
        reloaded = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                MODEL,
                revision=REVISION,
                dtype=torch.float32,
                attn_implementation="sdpa",
            ).to("cuda"),
            str(checkpoint),
        ).eval()
        with torch.no_grad():
            reload_logits = reloaded(**inputs).logits[:, -1].detach().cpu()
        torch.testing.assert_close(reload_logits, after_logits, rtol=1e-5, atol=1e-5)
        # Reset closes and flushes the last rollout for retained simulator evidence.
        client.request({"op": "reset", "policy_id": "smoke-finished"})
        recordings = list(data.rglob("*.jsonl"))
        assert recordings
        recorded_rows = [
            json.loads(line)
            for path in recordings
            for line in path.read_text().splitlines()
        ]
        recorded_episodes = {
            row["episode_id"] for row in recorded_rows if row.get("type") == "episode"
        }
        assert all(sample["episode_id"] in recorded_episodes for sample in samples)
        recorded_transitions = [
            row for row in recorded_rows if row.get("type") == "transition"
        ]
        assert len(recorded_transitions) == sum(sample["ticks"] for sample in samples)
        assert any(
            len(
                {
                    row["reward"]
                    for row in samples
                    if row["policy_id"] == policy and row["valid_action"]
                }
            )
            > 1
            for policy in {row["policy_id"] for row in samples}
        ), "No within-group game reward contrast"
        token_b = tokenizer.encode("B", add_special_tokens=False)
        assert len(token_b) == 1
        summary = {
            "ok": True,
            "model": MODEL,
            "revision": REVISION,
            "gpu": torch.cuda.get_device_name(),
            "cuda_capability": torch.cuda.get_device_capability(),
            "dtype": "float32",
            "versions": {
                n: importlib.metadata.version(n)
                for n in ("torch", "trl", "peft", "transformers")
            },
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "optimizer_steps": trainer.state.global_step,
            "sampled_rollouts": len(samples),
            "valid_actions": sum(s["valid_action"] for s in samples),
            "game_reward_values": sorted(
                {s["reward"] for s in samples if s["valid_action"]}
            ),
            "max_adapter_delta": delta,
            "max_policy_logit_delta": logit_delta,
            "reloaded_logits_match": True,
            "recorded_transitions": len(recorded_transitions),
            "throttle_probability_before": (before_logits / config.temperature)
            .softmax(-1)[0, token_b[0]]
            .item(),
            "throttle_probability_after": (after_logits / config.temperature)
            .softmax(-1)[0, token_b[0]]
            .item(),
            "trainable_parameters": sum(p.numel() for p in before.values()),
            "track_sha256": hashlib.sha256(
                (ROOT / "godot/data/track.json").read_bytes()
            ).hexdigest(),
            "lock_sha256": hashlib.sha256(
                (ROOT / "training/uv.lock").read_bytes()
            ).hexdigest(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.monotonic() - started,
            "train_metrics": trained.metrics,
            "scope": "One decision launch probe with Gemma3 270M, not E4B, camera training, racing skill or throughput",
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
