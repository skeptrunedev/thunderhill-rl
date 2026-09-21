"""Exercise TRL's native environment_factory against isolated Godot workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from check_parallel import worker

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class ThunderhillLaunch:
    """Two observed control decisions; each instance owns a distinct process."""

    def __init__(self, client, index, trace, step):
        self._client, self._index, self._trace, self._step = client, index, trace, step
        self._observation = None
        self._reward = 0.0
        self._calls = 0

    def _log(self, kind, **fields):
        self._trace.write(
            json.dumps(
                {
                    "type": kind,
                    "worker": self._index,
                    "episode_id": self._observation["episode_id"],
                    "policy_step": self._step(),
                    **fields,
                }
            )
            + "\n"
        )
        self._trace.flush()

    def _view(self):
        return {
            "tick": self._observation["tick"],
            "speed_m_s": self._observation["state"]["speed"],
            "done": self._observation["terminated"] or self._observation["truncated"],
        }

    def reset(self, **kwargs) -> str:
        self._observation = self._client.request(
            {"op": "reset", "policy_id": f"interactive-step-{self._step()}"}
        )
        if "error" in self._observation:
            raise RuntimeError(self._observation)
        self._reward, self._calls = 0.0, 0
        self._log("reset", observation=self._view())
        return (
            "You control a motorcycle stopped on a straight. Use the drive tool twice, "
            "making one call at a time and reading each resulting observation before the next call. "
            "Explore throttle values between 0 and 1. More forward progress earns more reward. "
            "Once done is true, answer Done and stop calling tools. Initial observation: "
            + json.dumps(self._view())
        )

    def drive(self, throttle: float) -> str:
        """Apply throttle for 0.1 simulated seconds and observe the updated motorcycle.

        Args:
            throttle: Throttle fraction between zero and one.

        Returns:
            Current tick, speed and whether the episode is finished.
        """
        if self._view()["done"]:
            raise ValueError("Episode finished; no further action allowed")
        if (
            isinstance(throttle, bool)
            or not isinstance(throttle, (float, int))
            or not math.isfinite(throttle)
            or not 0 <= throttle <= 1
        ):
            raise ValueError("Throttle must be a finite number between zero and one")
        before = self._view()
        result = self._client.request(
            {
                "op": "advance",
                "episode_id": self._observation["episode_id"],
                "expected_tick": before["tick"],
                "action_id": str(before["tick"]),
                "controls": {"throttle": throttle},
            }
        )
        if "error" in result or not result["rollout_valid"]:
            # Persist and flag infrastructure faults; the final validation also rejects them.
            self._log("infrastructure_failure", result=result)
            raise RuntimeError(result)
        self._observation = result
        self._calls += 1
        self._reward += sum(
            row["reward_components"]["legal_progress_m"]
            for row in result["transitions"]
        )
        self._log(
            "action",
            throttle=throttle,
            before=before,
            after=self._view(),
            cumulative_reward=self._reward,
        )
        return json.dumps(self._view())

    def get_reward(self) -> float:
        self._log(
            "reward", value=self._reward, calls=self._calls, observation=self._view()
        )
        return self._reward


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
            env = ThunderhillLaunch(
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
                max_completion_length=256,
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
        delta = max(
            (p.detach().cpu() - before[n]).abs().max().item()
            for n, p in trainer.model.named_parameters()
            if n in before
        )
        assert math.isfinite(delta) and delta > 0
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
            "excluded_environment_tokens": excluded,
            "max_adapter_delta": delta,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "seconds": time.monotonic() - started,
            "train_metrics": trained.metrics,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Two telemetry decisions per rollout with Qwen3 0.6B; not Gemma4 or camera training",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
