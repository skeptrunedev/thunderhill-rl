"""Evaluate a saved Gemma adapter with no teacher controller or fallback actions."""

import argparse
import hashlib
import json
import signal
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from agent_harness import ThunderhillEnv
from lap_audit import audit_lap
from lap_policy import RoadTelemetry, parse_action
from peft import PeftModel
from smoke_grpo import MODEL, REVISION, worker
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--godot", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-actions", type=int, default=9000)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
        ).to(args.device),
        str(args.adapter),
    ).eval()
    adapter_hash = hashlib.sha256(
        (args.adapter / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    road = RoadTelemetry()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    with ExitStack() as stack:
        client, data = stack.enter_context(
            worker(
                args.godot,
                out / "environment",
                90,
                (f"--agent-max-episode-ticks={args.max_actions * 12}",),
            )
        )
        trace = stack.enter_context((out / "harness.jsonl").open("w"))
        decisions = stack.enter_context((out / "decisions.jsonl").open("w"))
        env = ThunderhillEnv(
            client,
            0,
            trace,
            lambda: "lap-eval-" + adapter_hash[:12],
            road_telemetry=road,
        )
        env.reset()
        view = json.loads(env.observe())
        # Merge LoRA once to avoid separate adapter kernels on every decoded token.
        # Check numerical equivalence on the actual initial observation first.
        probe = tokenizer(road.prompt_features(view["road"]), return_tensors="pt").to(
            args.device
        )
        with torch.inference_mode():
            before_merge = model(**probe, logits_to_keep=1).logits.detach().clone()
            model = model.merge_and_unload(safe_merge=True).eval()
            after_merge = model(**probe, logits_to_keep=1).logits
            merge_error = (before_merge - after_merge).abs().max().item()
            if not torch.allclose(before_merge, after_merge, atol=1e-4, rtol=1e-4):
                raise RuntimeError(f"Adapter merge changed logits: {merge_error}")
        del before_merge, after_merge, probe
        episode = env._observation["episode_id"]
        reason = "action_budget"
        actions = 0
        for index in range(args.max_actions):
            if stop_requested:
                reason = "operator_stopped"
                break
            prompt = road.prompt_features(view["road"])
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    max_length=None,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            tokens = output[0, inputs["input_ids"].shape[1] :].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            row = {
                "action_index": index,
                "episode_id": episode,
                "prompt": prompt,
                "completion": text,
                "completion_ids": tokens,
                "adapter_sha256": adapter_hash,
            }
            try:
                controls = parse_action(text)
            except ValueError as error:
                row["error"] = str(error)
                decisions.write(json.dumps(row) + "\n")
                reason = "invalid_model_action"
                break
            view = json.loads(env.control_bike(view["observation_token"], **controls))
            actions += 1
            obs = env._observation
            row["tick"] = obs["tick"]
            row["controls"] = controls
            decisions.write(json.dumps(row) + "\n")
            decisions.flush()
            if index % 50 == 0:
                print(
                    json.dumps(
                        {
                            "actions": actions,
                            "progress": obs["track"]["progress"],
                            "speed": obs["state"]["speed"],
                            "lateral": obs["track"]["lateral_m"],
                            "wall_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
            if obs["terminated"] or obs["truncated"]:
                reason = (
                    obs.get("termination_reason")
                    or obs.get("truncation_reason")
                    or "finished"
                )
                break
            if not obs["track"]["lap_valid"]:
                reason = "track_limits"
                break
        final = env._observation
        client.request({"op": "reset", "policy_id": "evaluation-finished"})
        decisions.flush()
        audit = audit_lap(
            data.rglob("*.jsonl"),
            episode_id=episode,
            policy_id="interactive-step-lap-eval-" + adapter_hash[:12],
            track_sha256=road.track_sha256,
            final_observation=final,
            decisions=[
                json.loads(line)
                for line in (out / "decisions.jsonl").read_text().splitlines()
            ],
        )
        success = audit["success"]
        summary = {
            "success": success,
            "reason": reason,
            "model": MODEL,
            "device": args.device,
            "adapter_merged_for_inference": True,
            "merge_max_logit_error": merge_error,
            "adapter_sha256": adapter_hash,
            "teacher_used_at_inference": False,
            "observation_mode": "privileged_road_telemetry",
            "actions": actions,
            "sim_seconds": final["sim_time"],
            "wall_seconds": time.monotonic() - started,
            **audit,
            "final_observation": final,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in summary.items() if k != "final_observation"}),
            flush=True,
        )
        if not success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
