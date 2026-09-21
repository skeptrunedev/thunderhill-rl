"""Authority, observation freshness and failure contracts independent of a model."""

import io
import json
import unittest

from agent_harness import ThunderhillEnv


class Client:
    def __init__(self):
        self.requests = []
        self.tick = 0
        self.failure = None

    def request(self, command):
        self.requests.append(command)
        if self.failure:
            raise self.failure
        self.tick = 0 if command["op"] == "reset" else self.tick + 12
        result = {
            "episode_id": "fixture-episode",
            "tick": self.tick,
            "state": {"speed": self.tick / 10, "gear": 1, "lean": 0},
            "rollout_valid": True,
            "terminated": False,
            "truncated": self.tick >= 24,
        }
        if command["op"] == "advance":
            result["transitions"] = [
                {"reward_components": {"legal_progress_m": 0.01}} for _ in range(12)
            ]
        return result


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.client = Client()
        self.trace = io.StringIO()
        self.env = ThunderhillEnv(self.client, 0, self.trace, lambda: 7)
        self.env.reset()

    def test_observation_is_read_only_and_excludes_privileged_state(self):
        before = len(self.client.requests)
        first = json.loads(self.env.observe())
        self.assertEqual(self.env.observe(), self.env.observe())
        self.assertEqual(len(self.client.requests), before)
        self.assertEqual(
            set(first),
            {"tick", "speed_m_s", "gear", "lean_rad", "observation_token", "done"},
        )

    def test_fresh_token_required_and_complete_controls_forwarded(self):
        first = json.loads(self.env.observe())
        second = json.loads(
            self.env.control_bike(first["observation_token"], 0.4, -0.1, 0.2, 0.3, -1)
        )
        self.assertNotEqual(first["observation_token"], second["observation_token"])
        self.assertEqual(
            self.client.requests[-1]["controls"],
            {
                "throttle": 0.4,
                "steer": -0.1,
                "front_brake": 0.2,
                "rear_brake": 0.3,
                "shift": -1,
            },
        )
        count = len(self.client.requests)
        with self.assertRaisesRegex(ValueError, "Stale"):
            self.env.control_bike(first["observation_token"], 0.8)
        self.assertEqual(len(self.client.requests), count)
        final = json.loads(self.env.control_bike(second["observation_token"], 0.6))
        self.assertTrue(final["done"])
        with self.assertRaisesRegex(ValueError, "finished"):
            self.env.control_bike(final["observation_token"], 0.6)

    def test_invalid_controls_do_not_reach_simulator(self):
        token = json.loads(self.env.observe())["observation_token"]
        for controls in (
            {"throttle": True},
            {"throttle": float("nan")},
            {"throttle": 2},
            {"throttle": 0, "steer": -2},
            {"throttle": 0, "shift": 0.5},
            {"throttle": 0, "rear_brake": -1},
        ):
            with self.subTest(controls=controls):
                with self.assertRaises(ValueError):
                    self.env.control_bike(token, **controls)
                self.assertEqual(len(self.client.requests), 1)

    def test_transport_failure_cannot_become_training_reward(self):
        token = json.loads(self.env.observe())["observation_token"]
        self.client.failure = ConnectionError("injected disconnect")
        with self.assertRaises(ConnectionError):
            self.env.control_bike(token, 0.5)
        with self.assertRaisesRegex(RuntimeError, "Invalid rollout"):
            self.env.get_reward()
        rows = [json.loads(line) for line in self.trace.getvalue().splitlines()]
        self.assertEqual(
            sum(row["type"] == "infrastructure_failure" for row in rows), 1
        )
        self.assertFalse(any(row["type"] == "reward" for row in rows))

    def test_policy_cannot_change_fixed_duration_or_assistance(self):
        token = json.loads(self.env.observe())["observation_token"]
        for extra in (
            {"ticks": 9999},
            {"assist_enabled": False},
            {"auto_shift": False},
        ):
            with self.assertRaises(TypeError):
                self.env.control_bike(token, 0.5, **extra)
        self.assertEqual(len(self.client.requests), 1)
        self.assertEqual(self.client.requests[0]["policy_id"], "interactive-step-7")


if __name__ == "__main__":
    unittest.main()
