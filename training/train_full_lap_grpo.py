"""Bounded generations of complete, independently audited motorcycle episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
import random

import numpy as np
from pathlib import Path
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, set_seed

from batched_policy import BatchedPolicy, cuda_memory_evidence
from lap_episode import DEFAULT_STALL_CONFIG, FAILURE_PENALTY, PROGRESS_METERS_PER_REWARD, REWARD_VERSION, LapEpisode, episode_reward
from lap_audit import audit_lap
from lap_rollout import recorded_transitions
from experiment_tracking import ExperimentTracker
from model_runtime import FUNCTIONGEMMA_SPEC, GEMMA4_NATIVE_SPEC, QWEN27B_SPEC, QWEN_LORA_TARGET_MODULES, LORA_TARGET_MODULES, PolicyRoadTelemetry, inference_precision, load_base, read_spec, write_spec
from native_constraints import NativeToolConstraint
from trajectory_update import TrajectoryConfig, TrajectoryUpdater


def publish(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(adapter):
    return hashlib.sha256((Path(adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()


def collect(policy, road, spec, godot, directory, adapter_hash, generation, count, seconds, tracker=None,
            *, rollout_offset=0, rollout_total=None, evaluation_only=False):
    """Keep sampling weights frozen until every sampled and evaluation lane closes."""
    started = time.monotonic()
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=min(count + 1, 16)))
        episodes = [stack.enter_context(LapEpisode(
            godot=godot, output=directory / f"rollout-{index:04d}", road=road,
            adapter_sha256=adapter_hash, model=spec.model, revision=spec.revision,
            generation=generation, rollout=max(1, rollout_offset + index),
            rollout_count=max(1, rollout_total or count),
            time_budget_seconds=seconds, evaluation=evaluation_only or index == 0,
        )) for index in range(count + 1)]
        prompts = [episode.prompt() for episode in episodes]
        step = 0
        while any(not e.done for e in episodes):
            active = [i for i, e in enumerate(episodes) if not e.done]
            # Finished lanes retain their last prompt to keep compiled batch shapes
            # stable. Their unused samples never enter an episode or a loss mask.
            for i in active:
                prompts[i] = episodes[i].prompt()
            outputs = policy.generate(prompts, greedy_indices=(0,))
            def advance(i):
                row = outputs[i]
                episodes[i].apply(row["completion"], row["completion_ids"], row["prompt_ids"],
                                  behavior_logprobs=row["old_per_token_logps"])
                if episodes[i].done:
                    episodes[i].finish()
            list(pool.map(advance, active))
            step += 1
            if step % 100 == 0 or not any(not e.done for e in episodes):
                progress = dict(generation=generation, decision_steps=step,
                                active=sum(not e.done for e in episodes),
                                elapsed_seconds=time.monotonic() - started,
                                simulation_seconds=[e.observation["sim_time"] for e in episodes],
                                memory=cuda_memory_evidence(policy.model.device))
                publish(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                if tracker is not None:
                    tracker.log({"generation": generation, "collection/decision_steps": step,
                                 "collection/active_lanes": progress["active"],
                                 "collection/elapsed_seconds": progress["elapsed_seconds"]})
        summaries = [episode.finish() for episode in episodes]
        result = dict(generation=generation, adapter_sha256=adapter_hash,
                      elapsed_seconds=time.monotonic() - started,
                      evaluation=summaries[0], rollouts=summaries[1:])
        publish(directory / "collection.json", result)
        if tracker is not None:
            tracker.collection(result)
        if getattr(road, "native_tools", None) is not None and any(
            summary.get("reason") == "invalid_model_action" for summary in summaries
        ):
            raise ValueError("Native constrained generation produced an invalid tool call")
        training = []
        for episode, summary in ([] if evaluation_only else zip(episodes[1:], summaries[1:], strict=True)):
            if not summary["training_eligible"]:
                raise ValueError("Unusable episode cannot silently enter or leave the training group")
            training.append({"reward_components": summary["reward_components"], "decisions": [
                {"prompt_ids": r["prompt_ids"], "completion_ids": r["completion_ids"],
                 "old_per_token_logps": r["behavior_logprobs"]} for r in episode.records]})
        return training, result, prompts[0]



def collect_waves(policy, road, spec, godot, directory, adapter_hash, generation,
                  count, seconds, capacity, tracker=None, *, evaluation_seed=None):
    """Collect one frozen policy group with bounded concurrent simulator lanes.

    Evaluation uses a separate, repeatable sampling stream restored on exit.
    Its trajectories are deliberately never returned to the optimizer.
    """
    if count < 1 or capacity < 1:
        raise ValueError("Positive rollout count and wave capacity required")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    training, summaries, waves = [], [], []
    evaluation_only = evaluation_seed is not None
    with ExitStack() as stack:
        if evaluation_only:
            stack.enter_context(torch.random.fork_rng())
            torch.manual_seed(evaluation_seed)
        for offset in range(0, count, capacity):
            wave_training, result, prompt = collect(
                policy, road, spec, godot, directory / f"wave-{offset // capacity:04d}",
                adapter_hash, generation, min(capacity, count - offset), seconds,
                rollout_offset=offset, rollout_total=count, evaluation_only=evaluation_only)
            training.extend(wave_training)
            summaries.extend(result["rollouts"])
            waves.append(result)
    collection = dict(generation=generation, adapter_sha256=adapter_hash,
                      elapsed_seconds=time.monotonic() - started,
                      evaluation=waves[0]["evaluation"], rollouts=summaries, waves=waves)
    if evaluation_only:
        collection.update(evaluation_only=True, evaluation_seed=evaluation_seed)
    publish(directory / "collection.json", collection)
    if tracker is not None:
        tracker.collection(collection)
    return training, collection, prompt


def save_rng(directory):
    """Persist all sampling RNG streams without pickled numpy objects."""
    numpy_state = np.random.get_state()
    state = {"python": random.getstate(), "torch": torch.random.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}
    path = Path(directory) / "rng.pt"
    temporary = path.with_suffix(".pending")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_rng(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.random.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA RNG device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def resume_manifest(directory):
    """Resolve only fully published updates, rejecting ambiguous partial commits."""
    directory = Path(directory)
    manifest = json.loads((directory / "campaign.json").read_text())
    if manifest["complete"]:
        raise ValueError("Campaign is already complete")
    expected = {"reward_version": REWARD_VERSION, "progress_meters_per_reward": PROGRESS_METERS_PER_REWARD,
                "failure_penalty": FAILURE_PENALTY, "stall_config": asdict(DEFAULT_STALL_CONFIG),
                "supervised_training_performed": False, "training_method": "reinforcement_learning"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Resume campaign configuration mismatch: {key}")
    updates = manifest["generations"]
    initial = manifest.get("initial_generation", updates[0]["generation"] - 1 if updates else 0)
    if [row["generation"] for row in updates] != list(range(initial + 1, initial + len(updates) + 1)):
        raise ValueError("Resume requires contiguous committed generations")
    generation = initial + len(updates)
    checkpoint = directory / f"checkpoint-{generation:04d}" if updates else directory / "initial-adapter"
    actual = digest(checkpoint)
    expected_hash = updates[-1]["verification"]["adapter_sha256"] if updates else manifest["initial_adapter_sha256"]
    if actual != expected_hash:
        raise ValueError("Resume checkpoint hash mismatch")
    if updates:
        verification = json.loads((checkpoint / "verification.json").read_text())
        if verification.get("adapter_sha256") != actual or not verification.get("reloaded_logits_match"):
            raise ValueError("Resume checkpoint verification missing or invalid")
        if not (checkpoint / "optimizer.pt").is_file():
            raise ValueError("Resume requires persisted optimizer state")
    if (directory / f"checkpoint-{generation + 1:04d}").exists():
        raise ValueError("Uncommitted checkpoint exists; reconcile it before resuming")
    return manifest, checkpoint, initial


def load_collected_group(directory, adapter_hash, generation, count, road):
    """Reuse only frozen on policy trajectories backed by original recordings."""
    directory = Path(directory)
    collection = json.loads((directory / "collection.json").read_text())
    if collection.get("evaluation_only") or collection["adapter_sha256"] != adapter_hash or collection["generation"] != generation:
        raise ValueError("Collected group policy or generation mismatch")
    summaries, training, prompt = [], [], None
    for wave_index, wave in enumerate(collection["waves"]):
        if wave["adapter_sha256"] != adapter_hash or wave["generation"] != generation:
            raise ValueError("Collected wave policy or generation mismatch")
        for lane, summary in enumerate(wave["rollouts"], start=1):
            folder = directory / f"wave-{wave_index:04d}" / f"rollout-{lane:04d}"
            if json.loads((folder / "summary.json").read_text()) != summary:
                raise ValueError("Collected summary differs from persisted group")
            if not summary.get("training_eligible") or not summary.get("recording_provenance_verified"):
                raise ValueError("Collected episode not eligible for training")
            if summary["adapter_sha256"] != adapter_hash or summary["generation"] != generation:
                raise ValueError("Collected episode policy mismatch")
            job = json.loads((folder / summary["video_job"]).read_text())
            recording = folder / job["source"]
            if hashlib.sha256(recording.read_bytes()).hexdigest() != job["source_sha256"]:
                raise ValueError("Collected recording hash mismatch")
            rows = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
            if len(rows) != summary["actions"] or not rows:
                raise ValueError("Collected decision count mismatch")
            for row in rows:
                logps = row.get("behavior_logprobs", [])
                if (row["adapter_sha256"] != adapter_hash or not row["prompt_ids"] or
                    len(logps) != len(row["completion_ids"]) or not logps or
                    any(not math.isfinite(value) or value > 1e-5 for value in logps)):
                    raise ValueError("Collected behavior log probabilities or policy invalid")
            final = summary["final_observation"]
            audit = audit_lap([recording], episode_id=summary["episode_id"],
                              policy_id=final["policy_id"], track_sha256=road.track_sha256,
                              final_observation=final, decisions=rows, parse_completion=road.parse_completion)
            with recording.open() as stream:
                next(stream)
                progress = sum(row["reward_components"]["legal_progress_m"] for row in recorded_transitions(stream))
            reward = episode_reward(legal_progress_m=progress, track_length_m=road.length,
                sim_seconds=final["sim_time"], time_budget_seconds=summary["reward_components"]["time_budget_seconds"],
                success=audit["success"], failed=final["state"]["crashed"] or not final["track"]["lap_valid"],
                invalid_syntax=summary["reason"] == "invalid_model_action")
            if reward != summary["reward_components"]:
                raise ValueError("Collected reward differs from recording audit")
            training.append({"reward_components": reward, "decisions": [
                {"prompt_ids": row["prompt_ids"], "completion_ids": row["completion_ids"],
                 "old_per_token_logps": row["behavior_logprobs"]} for row in rows]})
            summaries.append(summary)
            prompt = rows[-1]["prompt"]
    if summaries != collection["rollouts"] or len(training) != count:
        raise ValueError("Collected group rollout count or ordering mismatch")
    return training, collection, prompt


def save_checkpoint(model, tokenizer, spec, updater, directory, prompt):
    """Round trip actual saved LoRA tensors and compare logits on the same base."""
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    write_spec(directory, spec)
    torch.save(updater.optimizer.state_dict(), directory / "optimizer.pt")
    model.eval()
    tokenize_options = {"add_special_tokens": False} if spec and spec.prompt_style.endswith("native_tools") else {}
    inputs = tokenizer(prompt, return_tensors="pt", **tokenize_options).to(model.device)
    with torch.inference_mode(), inference_precision(model):
        expected = model(**inputs, logits_to_keep=1).logits[:, -1].float().cpu()
    # A separately named adapter exercises PEFT's real deserializer, while keeping
    # the pretrained base and the active optimizer parameter identities unchanged.
    model.load_adapter(str(directory), adapter_name="roundtrip", is_trainable=False)
    try:
        model.set_adapter("roundtrip")
        with torch.inference_mode(), inference_precision(model):
            actual = model(**inputs, logits_to_keep=1).logits[:, -1].float().cpu()
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
    finally:
        model.set_adapter("default")
        model.delete_adapter("roundtrip")
    result = {"adapter_sha256": digest(directory), "reloaded_logits_match": True}
    publish(directory / "verification.json", result)
    save_rng(directory)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", type=Path)
    source.add_argument("--functiongemma", action="store_true", help="Start native gameplay RL from the pinned base with a fresh LoRA")
    source.add_argument("--qwen27b", action="store_true", help="Fresh Qwen27B LoRA trained only from native gameplay rewards")
    source.add_argument("--resume-campaign", type=Path, help="Continue an interrupted campaign in place from its verified model and optimizer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--initial-generation", type=int, default=2)
    parser.add_argument("--time-budget-seconds", type=float, default=900)
    parser.add_argument("--batch-candidates", default="4,8,16,32,64")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rollouts-per-generation", type=int)
    parser.add_argument("--evaluation-interval", type=int, default=0)
    parser.add_argument("--evaluation-rollouts", type=int, default=12)
    parser.add_argument("--evaluation-seed", type=int, default=1073)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--wandb-project", default="thunderhill-rl")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    resumed = None
    if args.resume_campaign:
        if args.output is not None and args.output.resolve() != args.resume_campaign.resolve():
            parser.error("Resume output must be the original campaign directory")
        args.output = args.resume_campaign
        resumed, args.adapter, args.initial_generation = resume_manifest(args.output)
        for option, key in (("generations", "generations_requested"), ("temperature", "temperature"),
                            ("time_budget_seconds", "time_budget_seconds"), ("rollouts_per_generation", "rollouts_per_generation"),
                            ("evaluation_interval", "evaluation_interval"), ("evaluation_rollouts", "evaluation_rollouts"),
                            ("evaluation_seed", "evaluation_seed"), ("smoke", "smoke_only")):
            setattr(args, option, resumed[key])
    elif args.output is None:
        parser.error("--output is required for a new campaign")
    if args.generations < 1 or args.time_budget_seconds <= 0:
        parser.error("Positive generation count and episode budget required")
    if args.rollouts_per_generation is not None and args.rollouts_per_generation < 2:
        parser.error("At least two sampled rollouts per generation required")
    if args.evaluation_interval < 0 or args.evaluation_rollouts < 1:
        parser.error("Nonnegative evaluation interval and positive evaluation rollouts required")
    candidates = [int(v) for v in args.batch_candidates.split(",")]
    if not candidates or any(v < 3 for v in candidates):
        parser.error("Each batch needs one evaluation lane and at least two sampled rollouts")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Temperature must be finite and positive")
    if resumed is None:
        args.output.mkdir(parents=True, exist_ok=False)
    set_seed(73)
    torch.set_num_threads(4)
    fresh_base = args.functiongemma or args.qwen27b
    spec = QWEN27B_SPEC if args.qwen27b else FUNCTIONGEMMA_SPEC if args.functiongemma else read_spec(args.adapter)
    if spec not in (GEMMA4_NATIVE_SPEC, FUNCTIONGEMMA_SPEC, QWEN27B_SPEC):
        raise ValueError("Full trajectory training requires a native tool model specification")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model if fresh_base else args.adapter,
        **({"revision": spec.revision} if fresh_base else {}), padding_side="left")
    road = PolicyRoadTelemetry(spec, tokenizer)
    if resumed is not None:
        for key, expected in {"model": spec.model, "revision": spec.revision,
                              "prompt_style": spec.prompt_style, "tools": road.native_tools.tools,
                              "action_version": road.native_tools.action_version,
                              "native_stop_token_id": road.native_tools.stop_token_id,
                              "constrained_sampling_and_training": True}.items():
            if resumed.get(key) != expected:
                raise ValueError(f"Resume model or native tool configuration mismatch: {key}")
    if fresh_base:
        model = get_peft_model(load_base(spec), LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0, task_type="CAUSAL_LM",
            target_modules=list(QWEN_LORA_TARGET_MODULES if spec == QWEN27B_SPEC else LORA_TARGET_MODULES)))
        args.adapter = args.output / "initial-adapter"
        model.save_pretrained(args.adapter)
        tokenizer.save_pretrained(args.adapter)
        write_spec(args.adapter, spec)
    else:
        model = PeftModel.from_pretrained(load_base(spec), str(args.adapter), is_trainable=True)
    if any(p.requires_grad and ("lora_" not in n or p.dtype != torch.float32)
           for n, p in model.named_parameters()):
        raise ValueError("Only FP32 LoRA parameters may be trainable")
    constraints = NativeToolConstraint(tokenizer, model.config.vocab_size, native_tools=road.native_tools)
    policy = BatchedPolicy(model, tokenizer, compile_inference=True,
                           compiled_prompt_length=1536 if spec == QWEN27B_SPEC else 1024, native_tools=road.native_tools,
                           constraints=constraints, temperature=args.temperature)
    seconds = 20.0 if args.smoke else args.time_budget_seconds
    generations = 1 if args.smoke else args.generations
    updater = TrajectoryUpdater(model, tokenizer, args.output / "updates",
                                TrajectoryConfig(max_actions=math.ceil(seconds * 10), temperature=args.temperature,
                                                 max_completion_length=road.native_tools.max_completion_length,
                                                 gradient_checkpointing=spec == QWEN27B_SPEC),
                                constraints=constraints)
    current_hash = digest(args.adapter)
    manifest = {"model": spec.model, "revision": spec.revision, "initial_adapter_sha256": current_hash,
                "training_method": "reinforcement_learning",
                "reward_version": REWARD_VERSION, "progress_meters_per_reward": PROGRESS_METERS_PER_REWARD,
                "failure_penalty": FAILURE_PENALTY, "seed": 73,
                "stall_config": asdict(DEFAULT_STALL_CONFIG),
                "evaluation_interval": args.evaluation_interval, "evaluation_rollouts": args.evaluation_rollouts,
                "evaluation_seed": args.evaluation_seed, "evaluations": [],
                "initialization": "fresh_base_lora" if fresh_base else "existing_adapter",
                "supervised_training_performed": False, "temperature": args.temperature,
                "gradient_checkpointing": spec == QWEN27B_SPEC,
                "lora_target_modules": list(QWEN_LORA_TARGET_MODULES if spec == QWEN27B_SPEC else LORA_TARGET_MODULES),
                "initial_generation": args.initial_generation, "generations_requested": generations, "time_budget_seconds": seconds,
                "smoke_only": args.smoke, "generations": [], "complete": False,
                "prompt_style": spec.prompt_style, "tools": road.native_tools.tools,
                "action_version": road.native_tools.action_version,
                "native_stop_token_id": road.native_tools.stop_token_id,
                "constrained_sampling_and_training": True}
    completed = 0
    resume_directory = None
    if resumed is not None:
        manifest = resumed
        completed = len(manifest["generations"])
        if completed:
            updater.optimizer.load_state_dict(torch.load(args.adapter / "optimizer.pt", map_location=model.device, weights_only=True))
        resume_directory = args.output / f"resume-{len(manifest.get('resumptions', [])) + 1:04d}"
        resume_directory.mkdir(exist_ok=False)
        next_collection = args.output / f"generation-{args.initial_generation + completed:04d}"
        rng_path = next_collection / "rng.pt" if (next_collection / "collection.json").exists() else args.adapter / "rng.pt"
        rng_present = rng_path.is_file()
        restart_seed = manifest["seed"] + 100000 + completed
        manifest.setdefault("resumptions", []).append({"completed_updates": completed,
            "checkpoint": str(args.adapter), "adapter_sha256": current_hash,
            "optimizer_restored": bool(completed), "rng_restored": rng_present,
            "rng_discontinuity": not rng_present, "restart_seed": None if rng_present else restart_seed,
            "rng_source": str(rng_path) if rng_present else None,
            "reused_collection": str(next_collection) if (next_collection / "collection.json").exists() else None})
    publish(args.output / "campaign.json", manifest)
    with ExperimentTracker(args.output / "tracking", manifest, mode=args.wandb_mode,
                           project=args.wandb_project, entity=args.wandb_entity, resume=resumed is not None) as tracker:
        # Obtain real prompt features from the identical simulator reset used below.
        # Even this short capacity probe is recorded and queued for video.
        with LapEpisode(godot=args.godot, output=(resume_directory or args.output) / "capacity-probe", road=road,
                        adapter_sha256=current_hash, model=spec.model, revision=spec.revision,
                        generation=args.initial_generation + completed, rollout=1, time_budget_seconds=.1) as probe:
            prompt = probe.prompt()
            if args.smoke:
                candidates = [value for value in candidates if value <= 8]
            profile = policy.profile([prompt], candidates=candidates)
            publish((resume_directory or args.output) / "inference-profile.json", profile)
            if profile["selected_batch_size"] is None:
                raise RuntimeError("No inference batch met validity and memory requirements")
            row = policy.generate([prompt], greedy_indices=(0,))[0]
            probe.apply(row["completion"], row["completion_ids"], row["prompt_ids"],
                        behavior_logprobs=row["old_per_token_logps"])
        if resumed is not None:
            if rng_present:
                restore_rng(rng_path)
            else:
                set_seed(restart_seed)
        # One lane evaluates greedily. All remaining lanes are sampled rollouts.
        capacity = profile["selected_batch_size"] - 1
        count = args.rollouts_per_generation or capacity
        manifest["rollouts_per_generation"] = count
        tracker.run.config.update({"rollouts_per_generation": count})
        publish(args.output / "campaign.json", manifest)
        print(json.dumps({"profile": profile, "rollouts_per_generation": count}), flush=True)
        def evaluate(generation):
            _, result, _ = collect_waves(policy, road, spec, args.godot,
                args.output / f"evaluation-{generation:04d}", current_hash, generation,
                args.evaluation_rollouts, seconds, capacity, tracker=tracker,
                evaluation_seed=args.evaluation_seed)
            manifest["evaluations"].append(result)
            publish(args.output / "campaign.json", manifest)

        if args.evaluation_interval:
            due = args.initial_generation + completed
            recorded = {row["generation"] for row in manifest["evaluations"]}
            if (completed % args.evaluation_interval == 0 or completed == generations) and due not in recorded:
                evaluate(due)
        for offset in range(completed, generations):
            generation = args.initial_generation + offset
            generation_directory = args.output / f"generation-{generation:04d}"
            if resumed is not None and (generation_directory / "collection.json").exists():
                episodes, collection, prompt = load_collected_group(generation_directory, current_hash, generation, count, road)
            else:
                episodes, collection, prompt = collect_waves(policy, road, spec, args.godot,
                    generation_directory, current_hash, generation, count, seconds, capacity, tracker=tracker)
            save_rng(generation_directory)
            # New rollout lengths and retained CUDA graphs change memory demand.
            # Requalify against each actual generation, allowing batches to grow or shrink.
            training_profile = updater.profile_microbatches(episodes)
            publish(args.output / f"generation-{generation:04d}" / "training-profile.json", training_profile)
            if offset == 0:
                publish(args.output / "training-profile.json", training_profile)
            audit = updater.update(episodes, generation=generation + 1)
            verification = save_checkpoint(model, tokenizer, spec, updater,
                args.output / f"checkpoint-{generation + 1:04d}", prompt)
            tracker.update(audit, verification)
            current_hash = verification["adapter_sha256"]
            manifest["generations"].append({"generation": generation + 1, "collection": collection,
                                            "update": audit, "verification": verification})
            publish(args.output / "campaign.json", manifest)
            print(json.dumps({"updated_generation": generation + 1, "audit": audit}), flush=True)
            if args.evaluation_interval and ((offset + 1) % args.evaluation_interval == 0 or offset + 1 == generations):
                evaluate(generation + 1)
        _, evaluation, _ = collect(policy, road, spec, args.godot,
            args.output / "final-evaluation", current_hash,
            args.initial_generation + generations, 0, seconds, tracker=tracker)
        manifest.update(complete=True, final_evaluation=evaluation)
        publish(args.output / "campaign.json", manifest)
        tracker.run.summary["campaign_complete"] = True


if __name__ == "__main__":
    main()
