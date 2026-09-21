"""TRL compatible bike tools. Policy inputs are distinct from trainer authority."""

from __future__ import annotations

import json
import math
import secrets


class ThunderhillEnv:
    """One simulator per rollout, with fresh observation receipts for actions.

    The current reward is a launch diagnostic, not a complete racing objective.
    The trainer owns worker creation, episode duration and checkpoint identity.
    """

    def __init__(self, client, index, trace, step):
        self._client, self._index, self._trace, self._step = client, index, trace, step
        self._observation = None
        self._reward = 0.0
        self._calls = 0
        self._receipt = ""
        self._fault = None

    def _log(self, kind, **fields):
        self._trace.write(
            json.dumps(
                {
                    "type": kind,
                    "worker": self._index,
                    "episode_id": self._observation.get("episode_id")
                    if self._observation
                    else None,
                    "policy_step": self._step(),
                    **fields,
                }
            )
            + "\n"
        )
        self._trace.flush()

    def _view(self):
        state = self._observation["state"]
        return {
            "tick": self._observation["tick"],
            "speed_m_s": state["speed"],
            "gear": state["gear"],
            "lean_rad": state["lean"],
            "observation_token": self._receipt,
            "done": self._observation["terminated"] or self._observation["truncated"],
        }

    def _request(self, request):
        if self._fault:
            raise RuntimeError("Rollout invalid after infrastructure failure")
        try:
            result = self._client.request(request)
            if "error" in result or not result.get("rollout_valid", False):
                raise RuntimeError(f"Invalid simulator response: {result}")
            for key in ("episode_id", "tick", "state", "terminated", "truncated"):
                if key not in result:
                    raise RuntimeError(f"Missing simulator field: {key}")
            return result
        except Exception as error:
            self._fault = str(error)
            self._log("infrastructure_failure", error=self._fault)
            raise

    def reset(self, **kwargs) -> str:
        self._fault = None
        self._observation = self._request(
            {"op": "reset", "policy_id": f"interactive-step-{self._step()}"}
        )
        self._reward, self._calls = 0.0, 0
        self._receipt = secrets.token_hex(4)
        self._log(
            "reset",
            observation=self._view(),
            observation_version="bike-telemetry-v1",
            reward_version="interactive-launch-v1",
        )
        return (
            "You control a stationary motorcycle on a straight. Explore throttle values between 0 and 1. "
            "More forward progress earns more reward. Call control_bike exactly ONCE in each response. "
            "Copy observation_token from the latest observation into the call. Never guess the next token. "
            "Wait for the tool result before choosing another action. Continue until done is true, then say Done. "
            "Steering assistance and automatic gears are enabled. Initial observation: "
            + json.dumps(self._view())
        )

    def observe(self) -> str:
        """Read the current observation without advancing simulation time.

        Returns:
            Telemetry and the token required for the next control action.
        """
        if self._fault:
            raise RuntimeError("Rollout invalid after infrastructure failure")
        return json.dumps(self._view())

    def control_bike(
        self,
        observation_token: str,
        throttle: float,
        steer: float = 0.0,
        front_brake: float = 0.0,
        rear_brake: float = 0.0,
        shift: int = 0,
    ) -> str:
        """Apply rider inputs for 0.1 simulated seconds, then return a new observation.

        Args:
            observation_token: Exact token from the latest observation. Never guess it.
            throttle: Throttle fraction from zero to one.
            steer: Normalized steering input from minus one to one, with balance assistance.
            front_brake: Front brake fraction from zero to one.
            rear_brake: Rear brake fraction from zero to one.
            shift: Gear change request, minus one, zero or one. Automatic shifting remains enabled.

        Returns:
            Updated telemetry, done flag and a fresh observation token.
        """
        if self._fault:
            raise RuntimeError("Rollout invalid after infrastructure failure")
        if observation_token != self._receipt:
            self._log("invalid_action", reason="stale_observation_token")
            raise ValueError(
                "Stale observation token. Call observe and use its exact token."
            )
        if self._view()["done"]:
            raise ValueError("Episode finished; no further action allowed")
        controls = {
            "throttle": throttle,
            "steer": steer,
            "front_brake": front_brake,
            "rear_brake": rear_brake,
            "shift": shift,
        }
        for name, value in controls.items():
            low = -1 if name in ("steer", "shift") else 0
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or not low <= value <= 1
            ):
                self._log("invalid_action", reason=f"invalid_{name}")
                raise ValueError(f"{name} must be finite and in [{low}, 1]")
        if shift != int(shift):
            self._log("invalid_action", reason="invalid_shift")
            raise ValueError("Shift must be minus one, zero or one")
        before = self._view()
        result = self._request(
            {
                "op": "advance",
                "episode_id": self._observation["episode_id"],
                "expected_tick": before["tick"],
                "action_id": self._receipt,
                "controls": controls,
            }
        )
        try:
            increment = sum(
                row["reward_components"]["legal_progress_m"]
                for row in result["transitions"]
            )
            if not math.isfinite(increment):
                raise ValueError("Nonfinite environment reward")
            self._observation = result
            self._receipt = secrets.token_hex(4)
            self._calls += 1
            self._reward += increment
            after = self._view()
        except Exception as error:
            self._fault = str(error)
            self._log("infrastructure_failure", error=self._fault)
            raise
        self._log(
            "action",
            controls=controls,
            before=before,
            after=after,
            cumulative_reward=self._reward,
        )
        return json.dumps(after)

    def get_reward(self) -> float:
        if self._fault:
            # TRL catches tool exceptions, but reward failure must abort the update.
            raise RuntimeError(
                f"Invalid rollout cannot receive a reward: {self._fault}"
            )
        self._log(
            "reward", value=self._reward, calls=self._calls, observation=self._view()
        )
        return self._reward
