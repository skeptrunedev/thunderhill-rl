"""Actual TRL first action Monte Carlo RL on a short moving road segment.

Only the sampled first control receives a policy gradient. Frozen prefix actions
and current policy greedy continuations are logged but never included in loss.
This is not full trajectory optimization or evidence of learning a complete lap.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import torch
from datasets import Dataset
from lap_policy import RoadTelemetry, parse_action
from lap_prefix import load_prefix
from lap_rollout import REWARD_VERSION, audit_rollout, physical_snapshot, rollout_reward
from peft import PeftModel
from smoke_grpo import MODEL, REVISION, worker
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


def checkpoint_hash(path):
    return hashlib.sha256(
        (Path(path) / "adapter_model.safetensors").read_bytes()
    ).hexdigest()


def request(client, payload):
    result = client.request(payload)
    if "error" in result or result.get("rollout_valid") is not True:
        raise RuntimeError(f"Simulator infrastructure failure: {result}")
    return result


def advance(client, observation, controls):
    return request(
        client,
        {
            "op": "advance",
            "episode_id": observation["episode_id"],
            "expected_tick": observation["tick"],
            "action_id": str(observation["tick"]),
            "controls": controls,
        },
    )


def finished(observation):
    return (
        observation["terminated"]
        or observation["truncated"]
        or not observation["track"]["lap_valid"]
    )


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=32,
                max_length=None,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        ids = output[0, inputs["input_ids"].shape[1] :].tolist()
        return tokenizer.decode(ids, skip_special_tokens=True), ids
    finally:
        model.train(was_training)


class MaskAuditTrainer(GRPOTrainer):
    """Retain the actual tensors TRL passes into its completion loss."""

    def __init__(self, *args, mask_trace, rollout_groups, **kwargs):
        self.mask_trace = mask_trace
        self.rollout_groups = rollout_groups
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        group = self.rollout_groups[-1]
        masks = result["completion_mask"].detach().cpu().tolist()
        ids = result["completion_ids"].detach().cpu().tolist()
        for row_ids, mask, sample in zip(ids, masks, group, strict=True):
            trained_ids = [
                token for token, keep in zip(row_ids, mask, strict=True) if keep
            ]
            if trained_ids != sample["sampled_completion_ids"]:
                raise ValueError(
                    "TRL loss includes tokens beyond the sampled first action"
                )
        row = {
            "step": self.state.global_step,
            "episodes": [sample["episode_id"] for sample in group],
            "prompt_ids": result["prompt_ids"].detach().cpu().tolist(),
            "prompt_mask": result["prompt_mask"].detach().cpu().tolist(),
            "completion_ids": ids,
            "completion_mask": masks,
            "advantages": result["advantages"].detach().cpu().tolist(),
            "prefix_and_continuation_tokens_in_loss": 0,
        }
        self.mask_trace.write(json.dumps(row) + "\n")
        self.mask_trace.flush()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument("--prefix-actions", type=int, default=40)
    parser.add_argument(
        "--prefix-decisions",
        type=Path,
        help="Replay this verified model evaluation prefix once per worker, then use worker snapshots",
    )
    parser.add_argument(
        "--prefix-adapter",
        type=Path,
        help="Source checkpoint for saved prefix, when different from --adapter",
    )
    parser.add_argument("--continuation-actions", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--validate-prefix-only",
        action="store_true",
        help="Verify saved prefix and four worker snapshots without loading model weights or using CUDA",
    )
    args = parser.parse_args()
    if (
        args.start_generation < 0
        or min(args.steps, args.prefix_actions, args.continuation_actions) < 1
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error(
            "Action budgets and steps must be positive; temperature and learning rate must be finite and positive"
        )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.validate_prefix_only and not args.prefix_decisions:
        parser.error("--validate-prefix-only requires --prefix-decisions")
    if not args.validate_prefix_only and not torch.cuda.is_available():
        raise RuntimeError("This training probe requires CUDA")
    set_seed(71)
    if not args.validate_prefix_only:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    road = RoadTelemetry()
    initial_hash = checkpoint_hash(args.adapter)
    prefix_hash = (
        checkpoint_hash(args.prefix_adapter) if args.prefix_adapter else initial_hash
    )
    if args.prefix_adapter and not args.prefix_decisions:
        parser.error("--prefix-adapter requires --prefix-decisions")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
    source_prefix, source_provenance = (
        load_prefix(
            args.prefix_decisions, args.prefix_actions, prefix_hash, road, tokenizer
        )
        if args.prefix_decisions
        else (None, None)
    )
    model = (
        None
        if args.validate_prefix_only
        else PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                MODEL,
                revision=REVISION,
                dtype=torch.float32,
                attn_implementation="sdpa",
            ).cuda(),
            str(args.adapter),
            is_trainable=True,
        )
    )
    samples, groups = [], []
    with ExitStack() as stack:
        max_ticks = (args.prefix_actions + 1 + args.continuation_actions) * 12
        workers = [
            stack.enter_context(
                worker(
                    args.godot,
                    out / f"environment-{index}",
                    90,
                    (f"--agent-max-episode-ticks={max_ticks}",),
                )
            )
            for index in range(4)
        ]
        trace = stack.enter_context((out / "rollouts.jsonl").open("w"))
        mask_trace = stack.enter_context((out / "loss-masks.jsonl").open("w"))
        snapshot_provenance = []
        if source_prefix is not None:

            def prepare_worker(index):
                client, _ = workers[index]
                obs = request(
                    client, {"op": "reset", "policy_id": "saved-model-prefix"}
                )
                original = physical_snapshot(obs)
                for frozen in source_prefix:
                    if (
                        obs["tick"] != frozen["before_tick"]
                        or road.prompt(obs) != frozen["prompt"]
                    ):
                        raise ValueError(
                            "Saved prefix replay prompt differs from source"
                        )
                    obs = advance(client, obs, frozen["controls"])
                    if obs["state"] != frozen["source_after_state"]:
                        raise ValueError(
                            "Saved prefix replay state differs from source"
                        )
                    for key in (
                        obs["track"].keys() & frozen["source_after_track"].keys()
                    ):
                        if obs["track"][key] != frozen["source_after_track"][key]:
                            raise ValueError(f"Saved prefix replay track {key} differs")
                    if finished(obs):
                        raise ValueError("Saved prefix replay failed")
                snapshot = client.request(
                    {
                        "op": "snapshot",
                        "episode_id": obs["episode_id"],
                        "expected_tick": obs["tick"],
                    }
                )
                if (
                    snapshot.get("version") != "worker-snapshot-v1"
                    or snapshot.get("source_episode_id") != obs["episode_id"]
                    or snapshot.get("source_tick") != obs["tick"]
                    or not snapshot.get("snapshot_id")
                ):
                    raise RuntimeError(f"Invalid worker snapshot response: {snapshot}")
                restored = request(
                    client,
                    {
                        "op": "reset",
                        "snapshot_id": snapshot["snapshot_id"],
                        "policy_id": "prefix-snapshot-validation",
                    },
                )
                if physical_snapshot(restored) != physical_snapshot(obs):
                    raise ValueError(
                        "Worker snapshot restore differs from captured branch"
                    )
                return original, obs, snapshot

            with ThreadPoolExecutor(max_workers=4) as pool:
                prepared = list(pool.map(prepare_worker, range(4)))
            reset_snapshot, observation, _ = prepared[0]
            for original, obs, snapshot in prepared:
                if original != reset_snapshot or physical_snapshot(
                    obs
                ) != physical_snapshot(observation):
                    raise ValueError("Worker prefix states are not identical")
                snapshot_provenance.append(snapshot)
            prefix = source_prefix
        else:
            client, _ = workers[0]
            observation = request(
                client, {"op": "reset", "policy_id": "frozen-model-prefix"}
            )
            reset_snapshot = physical_snapshot(observation)
            prefix = []
            for index in range(args.prefix_actions):
                prompt = road.prompt(observation)
                text, ids = generate(model, tokenizer, prompt)
                controls = parse_action(text)
                row = {
                    "phase": "prefix",
                    "before_tick": observation["tick"],
                    "prompt": prompt,
                    "completion": text,
                    "completion_ids": ids,
                    "controls": controls,
                    "adapter_sha256": prefix_hash,
                    "loss_mask": [0] * len(ids),
                }
                observation = advance(client, observation, controls)
                row["after_tick"] = observation["tick"]
                row["after_snapshot"] = physical_snapshot(observation)
                prefix.append(row)
                if finished(observation):
                    raise RuntimeError(
                        f"Frozen model prefix failed after {index + 1} controls"
                    )
        branch_snapshot = physical_snapshot(observation)
        if observation["state"]["speed"] <= 0:
            raise RuntimeError("Model prefix did not reach a moving state")
        prefix_tick = observation["tick"]
        branch_prompt = road.prompt(observation)
        (out / "frozen-prefix.json").write_text(
            json.dumps(
                {
                    "adapter_sha256": prefix_hash,
                    "source_provenance": source_provenance,
                    "worker_snapshots": snapshot_provenance,
                    "episode_id": observation["episode_id"],
                    "reset_snapshot": reset_snapshot,
                    "branch_snapshot": branch_snapshot,
                    "decisions": prefix,
                    "optimized_tokens": 0,
                },
                indent=2,
            )
            + "\n"
        )
        if args.validate_prefix_only:
            summary = {
                "ok": True,
                "prefix_validation_only": True,
                "model_weights_loaded": False,
                "cuda_training_run": False,
                "branch_tick": prefix_tick,
                "source_provenance": source_provenance,
                "identical_worker_snapshots": len(snapshot_provenance),
            }
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
            return

        def run_candidate(index, label, current_hash, generation, first=None):
            client, data = workers[index]
            display = {"model_name": MODEL, "generation": generation}
            if first is None:
                display["evaluation"] = True
            else:
                display.update(rollout_number=index + 1, rollout_count=4)
            reset_request = {
                "op": "reset",
                "policy_id": label,
                "policy_display": display,
            }
            if snapshot_provenance:
                reset_request["snapshot_id"] = snapshot_provenance[index]["snapshot_id"]
            obs = request(client, reset_request)
            episode_reset = physical_snapshot(obs)
            expected_reset = branch_snapshot if snapshot_provenance else reset_snapshot
            if episode_reset != expected_reset:
                raise ValueError(
                    "Candidate reset state differs from frozen prefix reset"
                )
            episode = obs["episode_id"]
            decisions, transitions = [], []
            for frozen in [] if snapshot_provenance else prefix:
                if road.prompt(obs) != frozen["prompt"]:
                    raise ValueError("Frozen prefix replay observation diverged")
                obs = advance(client, obs, frozen["controls"])
                if physical_snapshot(obs) != frozen["after_snapshot"]:
                    raise ValueError("Frozen prefix replay physics diverged")
                decisions.append(copy.deepcopy(frozen))
                transitions.extend(obs["transitions"])
            if (
                physical_snapshot(obs) != branch_snapshot
                or road.prompt(obs) != branch_prompt
            ):
                raise ValueError("Candidate branch state differs")
            invalid = False
            for action in range(1 + args.continuation_actions):
                prompt = road.prompt(obs)
                sampled = first is not None and action == 0
                text, ids = first if sampled else generate(model, tokenizer, prompt)
                decision = {
                    "phase": "sampled" if sampled else "continuation",
                    "before_tick": obs["tick"],
                    "prompt": prompt,
                    "completion": text,
                    "completion_ids": ids,
                    "adapter_sha256": current_hash,
                    "loss_mask": [int(sampled)] * len(ids),
                }
                try:
                    controls = parse_action(text)
                except ValueError as error:
                    decision["error"] = str(error)
                    decisions.append(decision)
                    invalid = True
                    break
                decision["controls"] = controls
                obs = advance(client, obs, controls)
                decision["after_tick"] = obs["tick"]
                decisions.append(decision)
                transitions.extend(obs["transitions"])
                if finished(obs):
                    break
            reward = rollout_reward(transitions, prefix_tick, invalid)
            stop_reason = (
                "invalid_model_action"
                if invalid
                else obs["termination_reason"]
                if obs["terminated"]
                else obs["truncation_reason"]
                if obs["truncated"]
                else "track_limits"
                if not obs["track"]["lap_valid"]
                else "trainer_action_horizon"
            )
            record = {
                "episode_id": episode,
                "policy_id": label,
                "policy_display": display,
                "worker": index,
                "prefix_adapter_sha256": prefix_hash,
                "current_adapter_sha256": current_hash,
                "prefix_tick": prefix_tick,
                "reset_snapshot": episode_reset,
                "snapshot_provenance": snapshot_provenance[index]
                if snapshot_provenance
                else None,
                "prefix_source_provenance": source_provenance,
                "branch_snapshot": branch_snapshot,
                "decisions": decisions,
                "sampled_completion_ids": first[1] if first else [],
                "sampled_completion": first[0] if first else None,
                "invalid_syntax": invalid,
                "rollout_stop_reason": stop_reason,
                "trainer_truncated": stop_reason == "trainer_action_horizon",
                "post_branch_action_budget": 1 + args.continuation_actions,
                "reward_components": reward,
                "final_observation": {
                    key: value for key, value in obs.items() if key != "transitions"
                },
            }
            request(client, {"op": "reset", "policy_id": "flush-finished-rollout"})
            record["recording_audit"] = audit_rollout(
                data.rglob("*.jsonl"), record, road.track_sha256
            )
            trace.write(json.dumps(record) + "\n")
            trace.flush()
            print(
                json.dumps(
                    {
                        "policy": label,
                        "reward": reward,
                        "audit": record["recording_audit"],
                    }
                ),
                flush=True,
            )
            return record

        baseline = run_candidate(
            0, "greedy-before", initial_hash, args.start_generation
        )

        def game_reward(prompts, completions, completion_ids, trainer_state, **kwargs):
            if len(completions) != 4 or any(
                prompt != branch_prompt for prompt in prompts
            ):
                raise ValueError(
                    "Expected exactly four candidates for the frozen branch state"
                )
            step = trainer_state.global_step
            snapshot_path = out / f"sampled-policy-step-{step}"
            model.save_pretrained(snapshot_path)
            tokenizer.save_pretrained(snapshot_path)
            current_hash = checkpoint_hash(snapshot_path)
            group = []
            for index, (text, ids) in enumerate(
                zip(completions, completion_ids, strict=True)
            ):
                if (
                    not isinstance(text, str)
                    or tokenizer.decode(ids, skip_special_tokens=True) != text
                ):
                    raise ValueError(
                        "Sampled completion text does not match generated tokens"
                    )
                record = run_candidate(
                    index,
                    f"grpo-step-{step}-candidate-{index}",
                    current_hash,
                    args.start_generation + step,
                    (text, ids),
                )
                record["optimizer_step"] = step
                samples.append(record)
                group.append(record)
            groups.append(group)
            return [row["reward_components"]["total"] for row in group]

        config = GRPOConfig(
            output_dir=str(out / "trainer"),
            max_steps=args.steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=32,
            learning_rate=args.learning_rate,
            beta=0.0,
            temperature=args.temperature,
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
        (out / "config.json").write_text(
            json.dumps(
                {
                    "method": "first action Monte Carlo GRPO with greedy current policy continuation",
                    "model": MODEL,
                    "revision": REVISION,
                    "initial_adapter_sha256": initial_hash,
                    "prefix_adapter_sha256": prefix_hash,
                    "prefix_source_provenance": source_provenance,
                    "snapshot_restarts": bool(snapshot_provenance),
                    "prefix_actions": args.prefix_actions,
                    "continuation_actions": args.continuation_actions,
                    "reward_version": REWARD_VERSION,
                    "track_sha256": road.track_sha256,
                    "trl_config": config.to_dict(),
                    "limitations": "One frozen moving state. Only first action tokens receive gradients. No teacher controls. Not full lap RL.",
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        trainer = MaskAuditTrainer(
            model=model,
            args=config,
            reward_funcs=game_reward,
            processing_class=tokenizer,
            train_dataset=Dataset.from_list(
                [{"prompt": branch_prompt}] * (4 * args.steps)
            ),
            mask_trace=mask_trace,
            rollout_groups=groups,
        )
        before = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        if not before or not all("lora_" in name for name in before):
            raise ValueError("Expected only trainable LoRA parameters")
        trained = trainer.train()
        if trainer.state.global_step != args.steps:
            raise ValueError("Training ended before requested optimizer steps")
        deltas = [
            (param.detach().cpu() - before[name]).abs().max().item()
            for name, param in model.named_parameters()
            if name in before
        ]
        if not torch.isfinite(torch.tensor(deltas)).all() or max(deltas) <= 0:
            raise ValueError("No finite nonzero LoRA update")
        delta = max(deltas)
        if not any(
            len(
                {
                    row["reward_components"]["legal_progress_m"]
                    for row in group
                    if not row["invalid_syntax"]
                }
            )
            > 1
            for group in groups
        ):
            raise ValueError(
                "No within group physical reward contrast among valid rollouts"
            )
        checkpoint = out / "adapter"
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        final_hash = checkpoint_hash(checkpoint)
        after = run_candidate(
            0,
            "greedy-after",
            final_hash,
            args.start_generation + trainer.state.global_step,
        )
        model.eval()
        inputs = tokenizer(branch_prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            expected_logits = (
                model(**inputs, logits_to_keep=1).logits[:, -1].detach().cpu()
            )
        model.cpu()
        reloaded = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(
                MODEL,
                revision=REVISION,
                dtype=torch.float32,
                attn_implementation="sdpa",
            ).cuda(),
            str(checkpoint),
        ).eval()
        with torch.inference_mode():
            reloaded_logits = (
                reloaded(**inputs, logits_to_keep=1).logits[:, -1].detach().cpu()
            )
        torch.testing.assert_close(
            expected_logits, reloaded_logits, rtol=1e-5, atol=1e-5
        )
        before_reward = baseline["reward_components"]["total"]
        after_reward = after["reward_components"]["total"]
        summary = {
            "ok": True,
            "model": MODEL,
            "revision": REVISION,
            "optimizer_steps": trainer.state.global_step,
            "sampled_rollouts": len(samples),
            "max_adapter_delta": delta,
            "reloaded_logits_match": True,
            "greedy_reward_before": before_reward,
            "greedy_reward_after": after_reward,
            "greedy_reward_delta": after_reward - before_reward,
            "greedy_return_improved": after_reward > before_reward,
            "greedy_controls_changed": [
                row.get("controls") for row in baseline["decisions"]
            ]
            != [row.get("controls") for row in after["decisions"]],
            "current_adapter_sha256": final_hash,
            "initial_adapter_sha256": initial_hash,
            "prefix_adapter_sha256": prefix_hash,
            "prefix_source_provenance": source_provenance,
            "branch_tick": prefix_tick,
            "greedy_stop_before": baseline["rollout_stop_reason"],
            "greedy_stop_after": after["rollout_stop_reason"],
            "reward_version": REWARD_VERSION,
            "train_metrics": trained.metrics,
            "groups": [[row["reward_components"] for row in group] for group in groups],
            "all_recording_audits_passed": True,
            "prefix_and_continuation_tokens_trained": 0,
            "gpu": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "trl", "peft")
            },
            "wall_seconds": time.monotonic() - started,
            "limitation": "One fixed moving segment, first action gradient only. An adapter update is not evidence of improved full lap driving.",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
