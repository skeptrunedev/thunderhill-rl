"""Bounded generations of complete, independently audited motorcycle episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer, set_seed

from batched_policy import BatchedPolicy
from lap_episode import LapEpisode
from model_runtime import GEMMA4_NATIVE_SPEC, PolicyRoadTelemetry, inference_precision, load_base, read_spec, write_spec
from native_constraints import NativeToolConstraint
from trajectory_update import TrajectoryConfig, TrajectoryUpdater


def publish(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(adapter):
    return hashlib.sha256((Path(adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()


def collect(policy, road, spec, godot, directory, adapter_hash, generation, count, seconds):
    """Keep sampling weights frozen until every sampled and evaluation lane closes."""
    started = time.monotonic()
    with ExitStack() as stack:
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=min(count + 1, 16)))
        episodes = [stack.enter_context(LapEpisode(
            godot=godot, output=directory / f"rollout-{index:04d}", road=road,
            adapter_sha256=adapter_hash, model=spec.model, revision=spec.revision,
            generation=generation, rollout=max(1, index), rollout_count=max(1, count),
            time_budget_seconds=seconds, evaluation=index == 0,
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
        summaries = [episode.finish() for episode in episodes]
        if getattr(road, "native_tools", None) is not None and any(
            summary.get("reason") == "invalid_model_action" for summary in summaries
        ):
            raise ValueError("Native constrained generation produced an invalid tool call")
        training = []
        for episode, summary in zip(episodes[1:], summaries[1:], strict=True):
            if not summary["training_eligible"]:
                raise ValueError("Unusable episode cannot silently enter or leave the training group")
            training.append({"reward_components": summary["reward_components"], "decisions": [
                {"prompt_ids": r["prompt_ids"], "completion_ids": r["completion_ids"],
                 "old_per_token_logps": r["behavior_logprobs"]} for r in episode.records]})
        result = dict(generation=generation, adapter_sha256=adapter_hash,
                      elapsed_seconds=time.monotonic() - started,
                      evaluation=summaries[0], rollouts=summaries[1:])
        publish(directory / "collection.json", result)
        return training, result, prompts[0]


def save_checkpoint(model, tokenizer, spec, updater, directory, prompt):
    """Round trip actual saved LoRA tensors and compare logits on the same base."""
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    write_spec(directory, spec)
    torch.save(updater.optimizer.state_dict(), directory / "optimizer.pt")
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--initial-generation", type=int, default=2)
    parser.add_argument("--time-budget-seconds", type=float, default=900)
    parser.add_argument("--batch-candidates", default="4,8,16,32,64")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.generations < 1 or args.time_budget_seconds <= 0:
        parser.error("Positive generation count and episode budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(73)
    torch.set_num_threads(4)
    spec = read_spec(args.adapter)
    if spec != GEMMA4_NATIVE_SPEC:
        raise ValueError("Full trajectory training requires a native tool adapter; migrate with train_lap_sft --native-tools first")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
    road = PolicyRoadTelemetry(spec, tokenizer)
    model = PeftModel.from_pretrained(load_base(spec), str(args.adapter), is_trainable=True)
    if any(p.requires_grad and ("lora_" not in n or p.dtype != torch.float32)
           for n, p in model.named_parameters()):
        raise ValueError("Only FP32 LoRA parameters may be trainable")
    constraints = NativeToolConstraint(tokenizer, model.config.vocab_size)
    policy = BatchedPolicy(model, tokenizer, compile_inference=True,
                           compiled_prompt_length=1024, native_tools=road.native_tools,
                           constraints=constraints)
    seconds = 20.0 if args.smoke else args.time_budget_seconds
    generations = 1 if args.smoke else args.generations
    updater = TrajectoryUpdater(model, tokenizer, args.output / "updates",
                                TrajectoryConfig(max_actions=math.ceil(seconds * 10),
                                                 max_completion_length=road.native_tools.max_completion_length),
                                constraints=constraints)
    current_hash = digest(args.adapter)
    manifest = {"model": spec.model, "revision": spec.revision, "initial_adapter_sha256": current_hash,
                "generations_requested": generations, "time_budget_seconds": seconds,
                "smoke_only": args.smoke, "generations": [], "complete": False,
                "prompt_style": spec.prompt_style, "tools": road.native_tools.tools,
                "native_stop_token_id": road.native_tools.stop_token_id,
                "constrained_sampling_and_training": True}
    publish(args.output / "campaign.json", manifest)
    # Obtain real prompt features from the identical simulator reset used below.
    # Even this short capacity probe is recorded and queued for video.
    with LapEpisode(godot=args.godot, output=args.output / "capacity-probe", road=road,
                    adapter_sha256=current_hash, model=spec.model, revision=spec.revision,
                    generation=args.initial_generation, rollout=1, time_budget_seconds=.1) as probe:
        prompt = probe.prompt()
        candidates = [int(v) for v in args.batch_candidates.split(",")]
        if args.smoke:
            candidates = [4, 8]
        profile = policy.profile([prompt], candidates=candidates)
        publish(args.output / "inference-profile.json", profile)
        if profile["selected_batch_size"] is None:
            raise RuntimeError("No inference batch met validity and memory requirements")
        row = policy.generate([prompt], greedy_indices=(0,))[0]
        probe.apply(row["completion"], row["completion_ids"], row["prompt_ids"],
                    behavior_logprobs=row["old_per_token_logps"])
    # One lane evaluates greedily. All remaining lanes are sampled rollouts.
    count = profile["selected_batch_size"] - 1
    manifest["rollouts_per_generation"] = count
    publish(args.output / "campaign.json", manifest)
    print(json.dumps({"profile": profile, "rollouts_per_generation": count}), flush=True)
    for offset in range(generations):
        generation = args.initial_generation + offset
        episodes, collection, prompt = collect(policy, road, spec, args.godot,
            args.output / f"generation-{generation:04d}", current_hash, generation, count, seconds)
        if offset == 0:
            training_profile = updater.profile_microbatches(episodes)
            publish(args.output / "training-profile.json", training_profile)
        audit = updater.update(episodes, generation=generation + 1)
        verification = save_checkpoint(model, tokenizer, spec, updater,
            args.output / f"checkpoint-{generation + 1:04d}", prompt)
        current_hash = verification["adapter_sha256"]
        manifest["generations"].append({"generation": generation + 1, "collection": collection,
                                        "update": audit, "verification": verification})
        publish(args.output / "campaign.json", manifest)
        print(json.dumps({"updated_generation": generation + 1, "audit": audit}), flush=True)
    _, evaluation, _ = collect(policy, road, spec, args.godot,
        args.output / "final-evaluation", current_hash,
        args.initial_generation + generations, 0, seconds)
    manifest.update(complete=True, final_evaluation=evaluation)
    publish(args.output / "campaign.json", manifest)


if __name__ == "__main__":
    main()
