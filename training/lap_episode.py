"""Full lap collection independent of the batched policy inference engine."""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from agent_harness import ThunderhillEnv
from lap_audit import audit_lap
from lap_policy import parse_action
from lap_rollout import recorded_transitions
from video_jobs import enqueue_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_parallel import worker

REWARD_VERSION = "audited-legal-progress-v2"
PROGRESS_METERS_PER_REWARD = 100.0
FAILURE_PENALTY = 0.2



@dataclass(frozen=True)
class StallConfig:
    """Integer physics tick windows; a stopped attempt remains RL experience."""
    grace_ticks: int = 600
    window_ticks: int = 600
    minimum_progress_m: float = 1.0

    def __post_init__(self):
        if (type(self.grace_ticks) is not int or self.grace_ticks < 0
                or type(self.window_ticks) is not int or self.window_ticks <= 0
                or not math.isfinite(self.minimum_progress_m) or self.minimum_progress_m <= 0):
            raise ValueError("Invalid stall monitor configuration")


DEFAULT_STALL_CONFIG = StallConfig()


class StallMonitor:
    """Check signed net legal progress after a complete post grace window.

    Retain the observation at or immediately before the window boundary. Normal
    twelve tick actions land exactly on the default six hundred tick boundary.
    For other cadences the diagnostic records the actual, slightly longer window.
    """
    def __init__(self, config=DEFAULT_STALL_CONFIG):
        self.config = config
        self.samples = deque()
        self.diagnostic = None

    def observe(self, tick, legal_distance):
        if (type(tick) is not int or tick < 0 or not math.isfinite(legal_distance)
                or (self.samples and tick <= self.samples[-1][0])):
            raise ValueError("Stall observations require increasing integer ticks and finite distance")
        self.samples.append((tick, legal_distance))
        cutoff = tick - self.config.window_ticks
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        start_tick, start_distance = self.samples[0]
        if (tick < self.config.grace_ticks + self.config.window_ticks
                or start_tick < self.config.grace_ticks or start_tick > cutoff):
            return False
        gain = legal_distance - start_distance
        stalled = gain < self.config.minimum_progress_m and not math.isclose(
            gain, self.config.minimum_progress_m, rel_tol=0, abs_tol=1e-9)
        self.diagnostic = {
            "start_tick": start_tick, "end_tick": tick,
            "window_ticks": tick - start_tick,
            "start_legal_distance_m": start_distance,
            "end_legal_distance_m": legal_distance,
            "legal_progress_m": gain,
            "minimum_progress_m": self.config.minimum_progress_m,
            "stalled": stalled,
        }
        return stalled

def episode_reward(*, legal_progress_m, track_length_m, sim_seconds,
                   time_budget_seconds, success, failed=False, invalid_syntax=False):
    """Signed legal progress with a failure cost equivalent to twenty meters.

    Success must come from audit_lap, never an unaudited model or simulator flag.
    Progress is capped at one track length; valid completion adds a bonus.
    Progress is signed net legal distance so reversing cannot farm rewards.
    """
    values = (legal_progress_m, track_length_m, sim_seconds, time_budget_seconds)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite episode reward input")
    if track_length_m <= 0 or time_budget_seconds <= 0 or sim_seconds < 0:
        raise ValueError("Invalid episode reward dimensions")
    if success and (failed or invalid_syntax or sim_seconds > time_budget_seconds + 1e-6):
        raise ValueError("Successful lap conflicts with failure or budget")
    progress_m = track_length_m if success else min(track_length_m, max(-track_length_m, legal_progress_m))
    result = {
        "version": REWARD_VERSION,
        "legal_progress_m": legal_progress_m,
        "track_length_m": track_length_m,
        "sim_seconds": sim_seconds,
        "time_budget_seconds": time_budget_seconds,
        "progress_fraction": progress_m / track_length_m,
        "progress_reward": progress_m / PROGRESS_METERS_PER_REWARD,
        "completion_bonus": 2.0 if success else 0.0,
        "speed_bonus": max(0.0, 1.0 - sim_seconds / time_budget_seconds) if success else 0.0,
        "failure_penalty": -FAILURE_PENALTY if failed else 0.0,
        "syntax_penalty": -1.0 if invalid_syntax else 0.0,
    }
    result["total"] = sum(result[k] for k in (
        "progress_reward", "completion_bonus", "speed_bonus", "failure_penalty", "syntax_penalty"))
    return result


class LapEpisode:
    """One isolated standing start, advanced by externally generated actions.

    Use with ExitStack. Every started episode is closed and queued on context exit,
    including inference failures. Infrastructure failures propagate and cannot
    receive training rewards. Each instance can run on a separate worker thread;
    calls on an individual instance must remain serialized.
    """
    def __init__(self, *, godot, output, road, adapter_sha256, model, revision,
                 generation, rollout, time_budget_seconds=900, evaluation=False, rollout_count=None,
                 worker_factory=worker, stall_config=DEFAULT_STALL_CONFIG):
        if len(adapter_sha256) != 64 or any(c not in "0123456789abcdef" for c in adapter_sha256):
            raise ValueError("Expected SHA256 adapter identity")
        ticks = round(time_budget_seconds * 120)
        if time_budget_seconds <= 0 or abs(ticks / 120 - time_budget_seconds) > 1e-8:
            raise ValueError("Episode budget must be positive whole physics ticks")
        self.godot, self.output, self.road = godot, Path(output).resolve(), road
        self.adapter_sha256, self.model, self.revision = adapter_sha256, model, revision
        self.generation, self.rollout = generation, rollout
        self.time_budget_seconds, self.max_ticks = time_budget_seconds, ticks
        self.display = {"model_name": model, "generation": generation}
        if evaluation:
            self.display["evaluation"] = True
        else:
            if rollout < 1 or (rollout_count is not None and rollout_count < rollout):
                raise ValueError("Rollout is one based and cannot exceed rollout_count")
            self.display.update(rollout_number=rollout, rollout_count=rollout_count or rollout)
        self.parse_completion = getattr(road, "parse_completion", parse_action)
        self.worker_factory = worker_factory
        self.stall_monitor = StallMonitor(stall_config)
        self.records = []
        self._previous_completion = None
        self._previous_features = None
        self.reason = None
        self.summary = None
        self._stack = ExitStack()
        self._started = False
        self._finished = False

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            self.client, self.data = self._stack.enter_context(self.worker_factory(
                self.godot, self.output / "environment", 90,
                (f"--agent-max-episode-ticks={self.max_ticks}",)))
            trace = self._stack.enter_context((self.output / "harness.jsonl").open("w"))
            self.decisions = self._stack.enter_context((self.output / "decisions.jsonl").open("w"))
            self.policy_step = f"lap-generation-{self.generation}-rollout-{self.rollout}-{self.adapter_sha256[:12]}"
            self.env = ThunderhillEnv(self.client, self.rollout, trace, lambda: self.policy_step,
                                      road_telemetry=self.road)
            self.env.reset(policy_display=self.display)
            self._started = True
            self.episode_id = self.env._observation["episode_id"]
            self.view = json.loads(self.env.observe())
            if self.env._observation["tick"] != 0 or self.env._observation["state"]["speed"] != 0:
                raise ValueError("Full lap requires standing start at tick zero")
            self.stall_monitor.observe(0, self.observation["track"]["legal_distance"])
            return self
        except BaseException:
            try:
                if self._started:
                    self.finish("infrastructure_failure", infrastructure_failure=True)
            finally:
                self._stack.close()
            raise

    @property
    def done(self):
        return self.reason is not None

    @property
    def observation(self):
        return self.env._observation

    def prompt(self):
        if self.done:
            raise ValueError("Episode finished")
        native = getattr(self.road, "native_tools", None)
        if native is not None and self._previous_completion is not None:
            # The receipt is an internal authorization detail attached by the
            # harness. Everything else is actual tool output from the simulator.
            response = {key: value for key, value in self.view.items()
                        if key != "observation_token"}
            return native.prompt(
                self.view["road"], previous_completion=self._previous_completion,
                previous_features=self._previous_features, tool_response=response,
            )
        return self.road.prompt_features(self.view["road"])

    def apply(self, completion, completion_ids, prompt_ids, *, behavior_logprobs=None):
        if self.done or self._finished:
            raise ValueError("Episode finished")
        row = {"action_index": len(self.records), "episode_id": self.episode_id,
               "prompt": self.prompt(), "prompt_ids": list(prompt_ids),
               "completion": completion, "completion_ids": list(completion_ids),
               "adapter_sha256": self.adapter_sha256, "before_tick": self.observation["tick"]}
        if behavior_logprobs is not None:
            if len(behavior_logprobs) != len(completion_ids) or not all(math.isfinite(x) for x in behavior_logprobs):
                raise ValueError("Behavior log probabilities must match completion tokens")
            row["behavior_logprobs"] = list(behavior_logprobs)
        try:
            controls = self.parse_completion(completion)
        except ValueError as error:
            row["error"] = str(error)
            self.reason = "invalid_model_action"
        else:
            before_features = self.view["road"]
            self.view = json.loads(self.env.control_bike(self.view["observation_token"], **controls))
            self._previous_features = before_features
            self._previous_completion = completion
            row.update(tick=self.observation["tick"], controls=controls)
            obs = self.observation
            if obs["terminated"] or obs["truncated"]:
                self.reason = obs.get("termination_reason") or obs.get("truncation_reason") or "finished"
            elif not obs["track"]["lap_valid"]:
                self.reason = "track_limits"
            elif self.stall_monitor.observe(obs["tick"], obs["track"]["legal_distance"]):
                self.reason = "stalled"
                row["stall_diagnostic"] = self.stall_monitor.diagnostic
        self.records.append(row)
        self.decisions.write(json.dumps(row, allow_nan=False) + "\n")
        self.decisions.flush()
        return row

    def finish(self, reason=None, *, infrastructure_failure=False):
        if self._finished:
            return self.summary
        self.reason = self.reason or reason or "collector_stopped"
        final = self.observation
        self.decisions.flush()
        # Reset flushes and closes the actual episode before publishing its hash.
        self.client.request({"op": "reset", "policy_id": "lap-collection-finished"})
        paths = list(self.data.rglob(f"{self.episode_id}.jsonl"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one closed lap recording")
        video = enqueue_video(self.output, paths[0], metadata={
            "kind": "full_lap_evaluation" if self.display.get("evaluation", False) else "full_lap_rollout",
            "adapter_sha256": self.adapter_sha256, "model": self.model, "revision": self.revision,
            "policy_display": self.display, "stop_reason": self.reason,
            "sim_seconds": final["sim_time"], "decisions": "decisions.jsonl",
            "stall_config": asdict(self.stall_monitor.config),
            "stall_diagnostic": self.stall_monitor.diagnostic})
        summary = {"episode_id": self.episode_id, "reason": self.reason,
                   "model": self.model, "revision": self.revision,
                   "generation": self.generation, "rollout": self.rollout,
                   "adapter_sha256": self.adapter_sha256,
                   "video_job": str(video.relative_to(self.output)),
                   "final_observation": final, "training_eligible": False,
                   "stall_config": asdict(self.stall_monitor.config),
                   "stall_diagnostic": self.stall_monitor.diagnostic}
        try:
            audit = audit_lap(paths, episode_id=self.episode_id,
                              policy_id="interactive-step-" + self.policy_step,
                              track_sha256=self.road.track_sha256,
                              final_observation=final, decisions=self.records,
                              parse_completion=self.parse_completion)
            summary.update(audit)
            if infrastructure_failure:
                summary["infrastructure_failure"] = True
            else:
                with paths[0].open() as source:
                    next(source)
                    legal_progress = sum(row["reward_components"]["legal_progress_m"]
                                         for row in recorded_transitions(source))
                summary["reward_components"] = episode_reward(
                    legal_progress_m=legal_progress, track_length_m=self.road.length,
                    sim_seconds=final["sim_time"], time_budget_seconds=self.time_budget_seconds,
                    success=audit["success"], failed=final["state"]["crashed"] or not final["track"]["lap_valid"],
                    invalid_syntax=self.reason == "invalid_model_action")
                summary["training_eligible"] = self.reason in {
                    "lap_completed", "crash", "episode_tick_limit", "track_limits", "invalid_model_action", "stalled"}
        except BaseException as error:
            summary["audit_error"] = str(error)
            raise
        finally:
            self.summary = summary
            self._finished = True
            (self.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary

    def abort(self, reason="infrastructure_failure"):
        """Archive a stopped attempt without producing a usable training reward."""
        return self.finish(reason, infrastructure_failure=True)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._started and not self._finished:
                self.finish("infrastructure_failure" if exc else None, infrastructure_failure=exc is not None)
        finally:
            self._stack.close()
        return False
