"""Exercise TRL's native environment_factory against isolated Godot workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from agent_harness import ThunderhillEnv
from check_parallel import worker

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class AuditedGRPO(GRPOTrainer):
    """Persist actual trainer tokens and loss masks without modifying training."""

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        mask = result.get("tool_mask")
        assert mask is not None, "Interactive generation did not return a tool mask"
        records = []
        for i in range(result["completion_ids"].shape[0]):
            ids = result["completion_ids"][i].detach().cpu().tolist()
            records.append(
                {
                    "policy_step": self.state.global_step,
                    "tokens": ids,
                    "text": self.processing_class.decode(ids),
                    "completion_mask": result["completion_mask"][i]
                    .detach()
                    .cpu()
                    .tolist(),
                    "tool_mask": mask[i].detach().cpu().tolist(),
                    "advantage": result["advantages"][i].item(),
                    "episode_id": self.environments[i]._observation["episode_id"],
                }
            )
        with (Path(self.args.output_dir).parent / "trainer-traces.jsonl").open(
            "a"
        ) as stream:
            for row in records:
                stream.write(json.dumps(row) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    set_seed(71)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
    ).cuda()
    with ExitStack() as stack:
        trace = stack.enter_context((out / "environment-traces.jsonl").open("w"))
        instances = []
        holder = {}

        def factory():
            index = len(instances)
            client, data = stack.enter_context(
                worker(
                    args.godot,
                    out / f"worker-{index}",
                    90,
                    ("--agent-max-episode-ticks=24",),
                )
            )
            env = ThunderhillEnv(
                client, index, trace, lambda: holder["trainer"].state.global_step
            )
            instances.append((env, data))
            return env

        trainer = AuditedGRPO(
            model=model,
            processing_class=tokenizer,
            environment_factory=factory,
            args=GRPOConfig(
                output_dir=str(out / "trainer"),
                max_steps=2,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_generations=4,
                max_completion_length=512,
                max_tool_calling_iterations=2,
                learning_rate=1e-4,
                temperature=1.1,
                top_p=1.0,
                top_k=0,
                beta=0.0,
                fp16=False,
                bf16=False,
                gradient_checkpointing=False,
                use_vllm=False,
                report_to="none",
                save_strategy="no",
                logging_steps=1,
                chat_template_kwargs={"enable_thinking": False},
                seed=71,
            ),
            peft_config=LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0,
                task_type="CAUSAL_LM",
            ),
        )
        holder["trainer"] = trainer
        before = {
            n: p.detach().cpu().clone()
            for n, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        assert before and all("lora_" in n for n in before)
        trained = trainer.train()
        deltas = [
            (p.detach().cpu() - before[n]).abs().max().item()
            for n, p in trainer.model.named_parameters()
            if n in before
        ]
        assert all(math.isfinite(value) for value in deltas), "Nonfinite adapter update"
        delta = max(deltas)
        assert delta > 0
        trainer.save_model(str(out / "adapter"))
        tokenizer.save_pretrained(str(out / "adapter"))
        for env, _ in instances:
            env._client.request({"op": "reset", "policy_id": "validation-finished"})
        events = [
            json.loads(line)
            for line in (out / "environment-traces.jsonl").read_text().splitlines()
        ]
        assert not any(row["type"] == "infrastructure_failure" for row in events)
        rewards = [row for row in events if row["type"] == "reward"]
        assert len(rewards) == 8
        assert all(row["calls"] >= 2 for row in rewards), (
            "A rollout failed to execute two decisions"
        )
        assert len({row["value"] for row in rewards}) > 1, (
            "No environment reward contrast"
        )
        traces = [
            json.loads(line)
            for line in (out / "trainer-traces.jsonl").read_text().splitlines()
        ]
        excluded = sum(
            sum(
                c and not t
                for c, t in zip(row["completion_mask"], row["tool_mask"], strict=True)
            )
            for row in traces
        )
        assert excluded > 0, "No environment tokens excluded from policy loss"
        assert all(
            any(
                c and t
                for c, t in zip(row["completion_mask"], row["tool_mask"], strict=True)
            )
            for row in traces
        )
        rows = [
            json.loads(line)
            for _, data in instances
            for path in data.rglob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        transitions = [row for row in rows if row["type"] == "transition"]
        assert len(transitions) == sum(row["observation"]["tick"] for row in rewards)
        assert len(traces) == len(rewards) == len({row["episode_id"] for row in traces})
        assert {row["episode_id"] for row in traces} == {
            row["episode_id"] for row in rewards
        }
        for trace_row in traces:
            episode = trace_row["episode_id"]
            actions = [
                row
                for row in events
                if row["episode_id"] == episode and row["type"] == "action"
            ]
            reward_row = next(row for row in rewards if row["episode_id"] == episode)
            recorded = [row for row in transitions if row["episode_id"] == episode]
            assert [row["tick"] for row in recorded] == list(
                range(1, reward_row["observation"]["tick"] + 1)
            )
            assert all(
                row["policy_id"] == f"interactive-step-{trace_row['policy_step']}"
                for row in recorded
            )
            assert math.isclose(
                sum(row["reward_components"]["legal_progress_m"] for row in recorded),
                reward_row["value"],
                abs_tol=1e-10,
            )
            masked_text = tokenizer.decode(
                [
                    token
                    for token, c, t in zip(
                        trace_row["tokens"],
                        trace_row["completion_mask"],
                        trace_row["tool_mask"],
                        strict=True,
                    )
                    if c and not t
                ]
            )
            generated_text = tokenizer.decode(
                [
                    token
                    for token, c, t in zip(
                        trace_row["tokens"],
                        trace_row["completion_mask"],
                        trace_row["tool_mask"],
                        strict=True,
                    )
                    if c and t
                ]
            )
            calls = [
                (match.start(), json.loads(match[1]))
                for match in re.finditer(
                    r"<tool_call>\s*(.*?)\s*</tool_call>", trace_row["text"], re.DOTALL
                )
            ]
            responses = [
                (match.end(), json.loads(match[1]))
                for match in re.finditer(
                    r"<tool_response>\s*(\{.*?\})\s*</tool_response>",
                    trace_row["text"],
                    re.DOTALL,
                )
            ]
            for action in actions[1:]:
                token = action["before"]["observation_token"]
                response_end = next(
                    end
                    for end, value in responses
                    if value.get("observation_token") == token
                )
                call_start = next(
                    start
                    for start, value in calls
                    if value.get("name") == "control_bike"
                    and value.get("arguments", {}).get("observation_token") == token
                )
                assert response_end < call_start, (
                    "Second action was chosen before its observation"
                )
                observation_json = json.dumps(action["before"])
                assert observation_json in masked_text, (
                    "Observation tokens were not fully excluded"
                )
                assert observation_json not in generated_text, (
                    "Observation leaked into policy loss"
                )
        summary = {
            "ok": True,
            "framework": "TRL environment_factory",
            "model": MODEL,
            "revision": REVISION,
            "optimizer_steps": trainer.state.global_step,
            "workers": len(instances),
            "rollouts": len(rewards),
            "calls_per_rollout": [row["calls"] for row in rewards],
            "rewards": [row["value"] for row in rewards],
            "recorded_transitions": len(transitions),
            "observed_action_order_verified": True,
            "per_episode_recording_and_mask_verified": True,
            "excluded_environment_tokens": excluded,
            "max_adapter_delta": delta,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "seconds": time.monotonic() - started,
            "train_metrics": trained.metrics,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "harness_sha256": hashlib.sha256(
                (Path(__file__).parent / "agent_harness.py").read_bytes()
            ).hexdigest(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "trl", "peft")
            },
            "lock_sha256": hashlib.sha256(
                (Path(__file__).parent / "uv.lock").read_bytes()
            ).hexdigest(),
            "scope": "Two telemetry decisions per rollout with Qwen3 0.6B; not Gemma4 or camera training",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
