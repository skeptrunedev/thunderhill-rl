"""Scalar W&B history for audited campaigns. Offline unless explicitly requested.

Import an existing campaign with:
  uv run --project training python training/experiment_tracking.py CAMPAIGN_JSON
Only scalar metrics and allowlisted model identity are logged, never recordings,
weights, prompts, tool responses, arbitrary CLI arguments, or credentials.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import json
import math
from pathlib import Path
from statistics import mean, median

CONFIG_KEYS = (
    "model", "revision", "initial_adapter_sha256", "generations_requested",
    "time_budget_seconds", "smoke_only", "prompt_style", "native_stop_token_id",
    "constrained_sampling_and_training", "rollouts_per_generation",
    "reward_version", "progress_meters_per_reward", "failure_penalty", "seed", "temperature",
    "training_method", "initialization", "supervised_training_performed",
    "stall_config", "evaluation_interval", "evaluation_rollouts", "evaluation_seed",
)
UPDATE_KEYS = (
    "loss", "episodes", "actions", "generated_tokens", "trained_tokens", "later_actions",
    "eos_tokens", "prompt_tokens_in_loss", "padding_tokens_in_loss", "fixed_normalizer",
    "max_behavior_logp_difference", "optimizer_steps", "gradient_norm", "parameter_delta_l1",
)


def episode_metrics(summary):
    """A failed attempt has a duration, never a lap time."""
    success = summary.get("success") is True and summary.get("recording_provenance_verified") is True
    invalid = summary["reason"] == "invalid_model_action"
    obs = summary["final_observation"]
    reward = summary.get("reward_components", {})
    actions = summary["actions"]
    metrics = {
        "completed": int(success), "crashed": int(obs["state"]["crashed"]),
        "stalled": int(summary["reason"] == "stalled"),
        "track_limits": int(summary["reason"] == "track_limits"),
        "invalid_call_count": int(invalid), "tool_call_count": actions + int(invalid),
        "episode_seconds": obs["sim_time"], "actions": actions,
        "offtrack_ticks": summary["offtrack_ticks"], "gates_passed": len(summary["gates"]),
    }
    for source, target in (("total", "reward"), ("legal_progress_m", "legal_progress_m"),
                           ("progress_fraction", "progress_fraction")):
        if source in reward:
            metrics[target] = reward[source]
    if success:
        metrics["lap_seconds"] = obs["sim_time"]
    return metrics


def collection_metrics(collection):
    result = {"generation": collection["generation"],
              "collection/wall_seconds": collection["elapsed_seconds"],
              "checkpoint/adapter_sha256": collection["adapter_sha256"]}
    result.update({"eval/" + key: value for key, value in episode_metrics(collection["evaluation"]).items()})
    rows = [episode_metrics(summary) for summary in collection["rollouts"]]
    result["rollout/count"] = len(rows)
    if rows:
        result.update({
            "rollout/completion_rate": mean(row["completed"] for row in rows),
            "rollout/crash_rate": mean(row["crashed"] for row in rows),
            "rollout/stall_rate": mean(row["stalled"] for row in rows),
            "rollout/track_limits_rate": mean(row["track_limits"] for row in rows),
            "rollout/invalid_episode_rate": mean(row["invalid_call_count"] for row in rows),
            "rollout/invalid_call_count": sum(row["invalid_call_count"] for row in rows),
            "rollout/tool_call_count": sum(row["tool_call_count"] for row in rows),
        })
        calls = result["rollout/tool_call_count"]
        if calls:
            result["rollout/invalid_call_rate"] = result["rollout/invalid_call_count"] / calls
        for key in ("reward", "legal_progress_m", "progress_fraction", "episode_seconds", "lap_seconds"):
            values = [row[key] for row in rows if key in row]
            if values:
                result[f"rollout/mean_{key}"] = mean(values)
                result[f"rollout/median_{key}"] = median(values)
        result["rollout/completed_count"] = sum(row["completed"] for row in rows)
    if collection.get("evaluation_only"):
        # These sampled episodes are held out from optimizer updates. Keep their
        # fixed-seed measurements separate from the changing training samples.
        result = {("heldout/" + key[len("rollout/"):] if key.startswith("rollout/") else
                   "heldout_greedy/" + key[len("eval/"):] if key.startswith("eval/") else key): value
                  for key, value in result.items()}
    return result


class ExperimentTracker(AbstractContextManager):
    def __init__(self, output, manifest, *, mode="offline", project="thunderhill-rl", entity=None, name=None,
                 resume=False):
        if mode not in {"offline", "online", "disabled"}:
            raise ValueError("Tracking mode must be offline, online, or disabled")
        import wandb
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        resume_options = {}
        if resume:
            previous = json.loads((output / "tracking.json").read_text())
            if mode != "online" or previous["mode"] != "online":
                raise ValueError("Tracking resume requires an existing online run")
            if previous["project"] != project or previous["entity"] != entity:
                raise ValueError("Tracking resume cannot change project or entity")
            resume_options = {"id": previous["run_id"], "resume": "must"}
        self.run = wandb.init(
            project=project, entity=entity, name=name or output.parent.name, mode=mode, dir=str(output),
            config={key: manifest[key] for key in CONFIG_KEYS if key in manifest},
            settings=wandb.Settings(disable_git=True, save_code=False, console="off"),
            **resume_options,
        )
        self.run.define_metric("generation")
        for namespace in ("rollout", "eval", "heldout", "heldout_greedy", "update", "collection", "checkpoint"):
            self.run.define_metric(namespace + "/*", step_metric="generation")
        self._history = (output / "metrics.jsonl").open("a" if resume else "w")
        (output / "tracking.json").write_text(json.dumps({
            "mode": mode, "run_id": self.run.id, "run_directory": self.run.dir,
            "project": project, "entity": entity,
        }, indent=2) + "\n")

    def log(self, metrics):
        if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
            raise ValueError("Nonfinite tracking metric")
        self.run.log(metrics)
        self._history.write(json.dumps(metrics, allow_nan=False) + "\n")
        self._history.flush()

    def collection(self, collection):
        self.log(collection_metrics(collection))

    def update(self, audit, verification):
        metrics = {"generation": audit["generation"],
                   "checkpoint/adapter_sha256": verification["adapter_sha256"],
                   "checkpoint/reloaded_logits_match": verification["reloaded_logits_match"]}
        metrics.update({"update/" + key: audit[key] for key in UPDATE_KEYS if key in audit})
        self.log(metrics)

    def __exit__(self, exc_type, exc, traceback):
        self._history.close()
        self.run.finish(exit_code=1 if exc_type else 0)
        return False


def import_campaign(path, output, **tracking):
    manifest = json.loads(Path(path).read_text())
    # Completed collections survive interrupted updates in separate files. Import
    # these too, without pretending that an interrupted update made a checkpoint.
    collections = {item["collection"]["generation"]: item["collection"]
                   for item in manifest["generations"]}
    for collection_path in sorted(Path(path).parent.glob("generation-*/collection.json")):
        row = json.loads(collection_path.read_text())
        collections.setdefault(row["generation"], row)
    events = [(generation, 1, "collection", row) for generation, row in collections.items()]
    events.extend((item["update"]["generation"], 0, "update", item)
                  for item in manifest["generations"])
    if "final_evaluation" in manifest:
        row = manifest["final_evaluation"]
        events.append((row["generation"], 2, "collection", row))
    for row in manifest.get("evaluations", []):
        events.append((row["generation"], 3, "collection", row))
    with ExperimentTracker(output, manifest, **tracking) as tracker:
        for _, _, kind, value in sorted(events, key=lambda item: item[:2]):
            if kind == "collection":
                tracker.collection(value)
            else:
                tracker.update(value["update"], value["verification"])
        tracker.run.summary["campaign_complete"] = manifest["complete"]
    return tracker.run.id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("offline", "online", "disabled"), default="offline")
    parser.add_argument("--project", default="thunderhill-rl")
    parser.add_argument("--entity")
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory to preserve previous imports")
    run_id = import_campaign(args.campaign, args.output, mode=args.mode,
                             project=args.project, entity=args.entity, name=args.name)
    print(json.dumps({"run_id": run_id, "output": str(args.output), "mode": args.mode}))


if __name__ == "__main__":
    main()
