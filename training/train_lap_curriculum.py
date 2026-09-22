"""Bounded local evaluation and TRL refinement of a learned motorcycle policy.

Only an audited track limits failure can start another training cycle. Every
trained adapter receives a fresh full lap evaluation, including the final cycle.
"""

import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lap_audit import audit_lap
from lap_policy import RoadTelemetry

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(path):
    return digest(Path(path) / "adapter_model.safetensors")


@dataclass
class Config:
    adapter: Path
    godot: str
    output: Path
    max_cycles: int = 3
    start_generation: int = 0
    max_actions: int = 9000
    steps: int = 10
    continuation_actions: int = 60
    temperature: float = 1.4
    learning_rate: float = 3e-5

    def validate(self):
        if (
            self.max_cycles < 0
            or self.start_generation < 0
            or min(self.max_actions, self.steps, self.continuation_actions) < 1
        ):
            raise ValueError(
                "Cycles must be nonnegative and action budgets and steps positive"
            )
        if not all(
            math.isfinite(x) and x > 0 for x in (self.temperature, self.learning_rate)
        ):
            raise ValueError(
                "Temperature and learning rate must be finite and positive"
            )


class ChildRunner:
    """Forward interrupts once and let the existing child clean up its workers."""

    def __call__(self, command, log):
        interrupted = False
        with log.open("x") as stream:
            child = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )

            def interrupt(signum, frame):
                nonlocal interrupted
                if not interrupted and child.poll() is None:
                    interrupted = True
                    child.send_signal(signal.SIGINT)

            previous = {
                sig: signal.signal(sig, interrupt)
                for sig in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                result = child.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            if interrupted:
                raise KeyboardInterrupt(
                    "Curriculum interrupted; child cleanup finished"
                )
            return result


def verify_evaluation(directory, summary, adapter_hash, track_hash):
    if (
        summary.get("recording_provenance_verified") is not True
        or summary.get("teacher_used_at_inference") is not False
        or summary.get("adapter_sha256") != adapter_hash
        or summary.get("compiled_inference") is not True
    ):
        raise ValueError("Evaluation provenance or compiled policy identity mismatch")
    decisions_path = directory / "decisions.jsonl"
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    final = summary["final_observation"]
    audit = audit_lap(
        summary["recordings"],
        episode_id=final["episode_id"],
        policy_id="interactive-step-lap-eval-" + adapter_hash[:12],
        track_sha256=track_hash,
        final_observation=final,
        decisions=decisions,
    )
    if any(summary.get(key) != value for key, value in audit.items()):
        raise ValueError(
            "Evaluation summary disagrees with independent recording audit"
        )
    return decisions


def run(config, runner=None):
    config.validate()
    runner = runner or ChildRunner()
    current = config.adapter.resolve()
    current_hash = checkpoint(current)
    out = config.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    track_hash = RoadTelemetry().track_sha256
    settings = {
        **asdict(config),
        "adapter": str(current),
        "output": str(out),
        "track_sha256": track_hash,
    }
    (out / "config.json").write_text(json.dumps(settings, indent=2, default=str) + "\n")
    lineage = [{"adapter": str(current), "sha256": current_hash, "parent_sha256": None}]

    def event(kind, **fields):
        row = {"time_unix": time.time(), "event": kind, **fields}
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def finish(reason, success=False):
        result = {
            "success": success,
            "reason": reason,
            "lineage": lineage,
            "final_adapter": str(current),
            "final_adapter_sha256": current_hash,
        }
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        event("finished", **result)
        return result

    def stage(name, script, arguments):
        directory = out / name
        command = [
            sys.executable,
            "-u",
            str(HERE / script),
            "--godot",
            config.godot,
            "--adapter",
            str(current),
            "--output",
            str(directory),
            *arguments,
        ]
        log = out / f"{name}.log"
        event(
            "stage_started",
            name=name,
            command=command,
            log=str(log),
            adapter_sha256=current_hash,
        )
        code = runner(command, log)
        event("stage_exited", name=name, returncode=code)
        if code not in (0, 1) or (script == "train_lap_grpo.py" and code != 0):
            raise RuntimeError(f"{name} failed with exit {code}; inspect {log}")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        event("stage_summary", name=name, summary_sha256=digest(summary_path))
        return directory, code, summary

    try:
        for cycle in range(config.max_cycles + 1):
            directory, code, evaluation = stage(
                f"cycle-{cycle:02d}-evaluation",
                "evaluate_lap.py",
                [
                    "--compile",
                    "--max-actions",
                    str(config.max_actions),
                    "--generation",
                    str(config.start_generation + cycle * config.steps),
                ],
            )
            decisions = verify_evaluation(
                directory, evaluation, current_hash, track_hash
            )
            if evaluation["success"] is True:
                if code != 0:
                    raise ValueError("Successful lap returned nonzero exit status")
                return finish("lap_completed", True)
            if code != 1 or evaluation["success"] is not False:
                raise ValueError(
                    "Failed evaluation must explicitly report failure and exit one"
                )
            reason = evaluation["reason"]
            if reason not in {
                "track_limits",
                "invalid_model_action",
                "action_budget",
                "episode_tick_limit",
                "crash",
                "operator_stopped",
            }:
                raise ValueError(f"Unknown failed evaluation reason: {reason}")
            if reason != "track_limits":
                return finish(f"evaluation_stopped:{reason}")
            final = evaluation["final_observation"]
            if final["track"]["lap_valid"] or final["terminated"] or final["truncated"]:
                raise ValueError("Track limits reason contradicts final recorded state")
            actions = evaluation["actions"]
            if (
                not isinstance(actions, int)
                or isinstance(actions, bool)
                or len(decisions) != actions
            ):
                raise ValueError(
                    "Executed action count differs from complete decisions trace"
                )
            if actions < 26:
                return finish("insufficient_valid_prefix")
            if cycle == config.max_cycles:
                return finish("maximum_training_cycles_reached")
            prefix_count = actions - 25
            if prefix_count > len(decisions) or any(
                "error" in row or row.get("tick") != (i + 1) * 12
                for i, row in enumerate(decisions[:prefix_count])
            ):
                raise ValueError("Prefix exceeds complete executed action history")
            trained_dir, _, trained = stage(
                f"cycle-{cycle:02d}-training",
                "train_lap_grpo.py",
                [
                    "--start-generation",
                    str(config.start_generation + cycle * config.steps),
                    "--prefix-decisions",
                    str(directory / "decisions.jsonl"),
                    "--prefix-actions",
                    str(prefix_count),
                    "--continuation-actions",
                    str(config.continuation_actions),
                    "--steps",
                    str(config.steps),
                    "--temperature",
                    str(config.temperature),
                    "--learning-rate",
                    str(config.learning_rate),
                ],
            )
            next_adapter = trained_dir / "adapter"
            next_hash = checkpoint(next_adapter)
            if (
                trained.get("ok") is not True
                or trained.get("all_recording_audits_passed") is not True
                or trained.get("reloaded_logits_match") is not True
                or trained.get("optimizer_steps") != config.steps
                or trained.get("initial_adapter_sha256") != current_hash
                or trained.get("prefix_adapter_sha256") != current_hash
                or trained.get("current_adapter_sha256") != next_hash
                or trained.get("prefix_and_continuation_tokens_trained") != 0
                or next_hash == current_hash
            ):
                raise ValueError(
                    "Training summary failed checkpoint lineage or audit checks"
                )
            source = trained.get("prefix_source_provenance", {})
            if (
                source.get("all_source_controls_and_states_verified") is not True
                or source.get("actions") != prefix_count
                or source.get("decision_sha256")
                != digest(directory / "decisions.jsonl")
                or source.get("adapter_sha256") != current_hash
            ):
                raise ValueError(
                    "Training prefix provenance differs from evaluated decisions"
                )
            lineage.append(
                {
                    "adapter": str(next_adapter),
                    "sha256": next_hash,
                    "parent_sha256": current_hash,
                    "training": str(trained_dir),
                    "source_evaluation": str(directory),
                    "prefix_actions": prefix_count,
                }
            )
            current, current_hash = next_adapter, next_hash
            event("checkpoint_verified", **lineage[-1])
    except KeyboardInterrupt:
        finish("operator_interrupted")
        raise
    except Exception as error:
        finish(f"infrastructure_or_provenance_error:{type(error).__name__}:{error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--start-generation", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=9000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--continuation-actions", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    result = run(Config(**vars(parser.parse_args())))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
