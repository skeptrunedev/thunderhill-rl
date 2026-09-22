"""Bounded generations of complete, independently audited motorcycle episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer, set_seed

from batched_policy import BatchedPolicy
from lap_episode import DEFAULT_STALL_CONFIG, FAILURE_PENALTY, PROGRESS_METERS_PER_REWARD, REWARD_VERSION, LapEpisode
from experiment_tracking import ExperimentTracker
from model_runtime import FUNCTIONGEMMA_SPEC, GEMMA4_NATIVE_SPEC, LORA_TARGET_MODULES, PolicyRoadTelemetry, inference_precision, load_base, read_spec, write_spec
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
                                simulation_seconds=[e.observation["sim_time"] for e in episodes])
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
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", type=Path)
    source.add_argument("--functiongemma", action="store_true", help="Start native gameplay RL from the pinned base with a fresh LoRA")
    parser.add_argument("--output", type=Path, required=True)
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
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(73)
    torch.set_num_threads(4)
    spec = FUNCTIONGEMMA_SPEC if args.functiongemma else read_spec(args.adapter)
    if spec not in (GEMMA4_NATIVE_SPEC, FUNCTIONGEMMA_SPEC):
        raise ValueError("Full trajectory training requires a native tool model specification")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model if args.functiongemma else args.adapter,
        **({"revision": spec.revision} if args.functiongemma else {}), padding_side="left")
    road = PolicyRoadTelemetry(spec, tokenizer)
    if args.functiongemma:
        model = get_peft_model(load_base(spec), LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0, task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGET_MODULES)))
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
                           compiled_prompt_length=1024, native_tools=road.native_tools,
                           constraints=constraints, temperature=args.temperature)
    seconds = 20.0 if args.smoke else args.time_budget_seconds
    generations = 1 if args.smoke else args.generations
    updater = TrajectoryUpdater(model, tokenizer, args.output / "updates",
                                TrajectoryConfig(max_actions=math.ceil(seconds * 10), temperature=args.temperature,
                                                 max_completion_length=road.native_tools.max_completion_length),
                                constraints=constraints)
    current_hash = digest(args.adapter)
    manifest = {"model": spec.model, "revision": spec.revision, "initial_adapter_sha256": current_hash,
                "training_method": "reinforcement_learning",
                "reward_version": REWARD_VERSION, "progress_meters_per_reward": PROGRESS_METERS_PER_REWARD,
                "failure_penalty": FAILURE_PENALTY, "seed": 73,
                "stall_config": asdict(DEFAULT_STALL_CONFIG),
                "evaluation_interval": args.evaluation_interval, "evaluation_rollouts": args.evaluation_rollouts,
                "evaluation_seed": args.evaluation_seed, "evaluations": [],
                "initialization": "fresh_base_lora" if args.functiongemma else "existing_adapter",
                "supervised_training_performed": False, "temperature": args.temperature,
                "generations_requested": generations, "time_budget_seconds": seconds,
                "smoke_only": args.smoke, "generations": [], "complete": False,
                "prompt_style": spec.prompt_style, "tools": road.native_tools.tools,
                "action_version": road.native_tools.action_version,
                "native_stop_token_id": road.native_tools.stop_token_id,
                "constrained_sampling_and_training": True}
    publish(args.output / "campaign.json", manifest)
    with ExperimentTracker(args.output / "tracking", manifest, mode=args.wandb_mode,
                           project=args.wandb_project, entity=args.wandb_entity) as tracker:
        # Obtain real prompt features from the identical simulator reset used below.
        # Even this short capacity probe is recorded and queued for video.
        with LapEpisode(godot=args.godot, output=args.output / "capacity-probe", road=road,
                        adapter_sha256=current_hash, model=spec.model, revision=spec.revision,
                        generation=args.initial_generation, rollout=1, time_budget_seconds=.1) as probe:
            prompt = probe.prompt()
            if args.smoke:
                candidates = [value for value in candidates if value <= 8]
            profile = policy.profile([prompt], candidates=candidates)
            publish(args.output / "inference-profile.json", profile)
            if profile["selected_batch_size"] is None:
                raise RuntimeError("No inference batch met validity and memory requirements")
            row = policy.generate([prompt], greedy_indices=(0,))[0]
            probe.apply(row["completion"], row["completion_ids"], row["prompt_ids"],
                        behavior_logprobs=row["old_per_token_logps"])
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
            evaluate(args.initial_generation)
        for offset in range(generations):
            generation = args.initial_generation + offset
            episodes, collection, prompt = collect_waves(policy, road, spec, args.godot,
                args.output / f"generation-{generation:04d}", current_hash, generation, count, seconds,
                capacity, tracker=tracker)
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
