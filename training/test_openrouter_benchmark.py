"""Native API tool evaluation without model downloads, credentials or training."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError, URLError

from openrouter_benchmark import (APIError, APINativeTools, OpenRouterClient, aggregate,
                                 canonical, configuration, run_episode, strict_json,
                                 select_endpoint, validate_models)


def message(arguments=None):
    if arguments is None:
        arguments = {"steer_milli": 0, "throttle_percent": 0,
                     "front_brake_percent": 100, "rear_brake_percent": 0}
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call-test", "type": "function",
        "function": {"name": "control_bike", "arguments": json.dumps(arguments)}}]}


def response(tool_message=None, provider="Test Provider"):
    return {"id": "response-test", "model": "test/model", "provider": provider,
            "choices": [{"message": message() if tool_message is None else tool_message,
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30}}


ENTRY = {"id": "test/model", "supported_parameters": [
    "tools", "tool_choice", "temperature", "max_tokens", "seed", "reasoning"]}


class NativeToolTests(unittest.TestCase):
    def setUp(self):
        self.tools = APINativeTools()

    def test_native_integer_controls(self):
        actual = self.tools.parse_completion(canonical(message()))
        self.assertEqual(actual, {"steer": 0, "throttle": 0,
                                  "front_brake": 1, "rear_brake": 0, "shift": 0})

    def test_invalid_arguments_never_repaired(self):
        valid = json.loads(message()["tool_calls"][0]["function"]["arguments"])
        for name, value in (("throttle_percent", True), ("steer_milli", 1001),
                            ("front_brake_percent", -1), ("throttle_percent", 1.5),
                            ("steer_milli", "0"), ("throttle_percent", None)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                self.tools.parse_completion(canonical(message({**valid, name: value})))
        for args in ({**valid, "shift": 0}, {k: v for k, v in valid.items() if k != "steer_milli"}):
            with self.assertRaises(ValueError):
                self.tools.parse_completion(canonical(message(args)))

    def test_duplicate_json_and_nonfinite_rejected(self):
        bad = message()
        bad["tool_calls"][0]["function"]["arguments"] = (
            '{"steer_milli":0,"steer_milli":100,"throttle_percent":0,'
            '"front_brake_percent":100,"rear_brake_percent":0}')
        with self.assertRaises(ValueError):
            self.tools.parse_completion(canonical(bad))
        for text in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_multiple_missing_and_unknown_calls(self):
        multiple = message()
        multiple["tool_calls"] *= 2
        unknown = message()
        unknown["tool_calls"][0]["function"]["name"] = "steer"
        no_id = message()
        no_id["tool_calls"][0].pop("id")
        for bad in (multiple, unknown, no_id, {"role": "assistant", "content": "go"}, None, []):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.tools.parse_completion(canonical(bad))

    def test_rolling_feedback_preserves_native_message(self):
        previous = message()
        previous["reasoning_details"] = [{"type": "reasoning.text", "text": "Choose control"}]
        features = {"speed": 2}
        result = {"tick": 12, "road": features}
        messages = strict_json(self.tools.prompt(features, previous_completion=canonical(previous),
                                                previous_features={"speed": 0}, tool_response=result))
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[2], previous)
        self.assertEqual(messages[3]["tool_call_id"], "call-test")
        self.assertEqual(strict_json(messages[3]["content"]), result)
        with self.assertRaises(ValueError):
            self.tools.prompt(features, previous_completion=canonical(previous),
                              previous_features={}, tool_response={"road": {}})

    def test_catalog_and_config_are_explicit(self):
        self.assertEqual(validate_models({"data": [ENTRY]}, ["test/model"]), {"test/model": ENTRY})
        with self.assertRaises(ValueError):
            validate_models({"data": [ENTRY]}, ["missing"])
        with self.assertRaises(ValueError):
            validate_models({"data": [{**ENTRY, "supported_parameters": ["tools"]}]}, ["test/model"])
        entry = {**ENTRY, "supported_parameters": [x for x in ENTRY["supported_parameters"] if x != "seed"]}
        config = configuration("test/model", entry, seconds=30, temperature=.6, seed=1073,
                               max_tokens=512, reasoning="disabled")
        self.assertIsNone(config["seed"])
        self.assertEqual(config["requested_seed"], 1073)
        self.assertEqual(config["reasoning"], {"enabled": False})
        self.assertFalse(config["supervised_training_performed"])

    def test_endpoint_capabilities_intersect_and_pin_exact_slug(self):
        config = configuration("test/model", ENTRY, seconds=30, temperature=.6, seed=1073,
                               max_tokens=512, reasoning="disabled")
        forced = {"tag": "provider/bf16", "provider_name": "Provider Display Name",
                  "quantization": "bf16", "supported_parameters": ENTRY["supported_parameters"],
                  "supports_tool_choice": {"function": True, "required": False},
                  "max_completion_tokens": 4096}
        no_seed = {**forced, "tag": "other/fp8", "quantization": "fp8",
                   "supported_parameters": [x for x in ENTRY["supported_parameters"] if x != "seed"],
                   "supports_tool_choice": {"required": True, "function": True}}
        result = select_endpoint(config, {"data": {"endpoints": [no_seed, forced]}})
        self.assertEqual(result["provider"], {"only": ["provider/bf16"],
                         "require_parameters": True, "allow_fallbacks": False})
        self.assertEqual(result["tool_choice"], {"type": "function", "function": {"name": "control_bike"}})
        self.assertEqual(result["seed"], 1073)
        with self.assertRaises(ValueError):
            select_endpoint(config, {"data": {"endpoints": [no_seed]}})
        with self.assertRaises(ValueError):
            select_endpoint(config, {"data": {"endpoints": [{**forced,
                "supports_tool_choice": {"required": False, "function": False}}]}})


class ClientTests(unittest.TestCase):
    def test_native_request_and_response(self):
        seen = []
        def opener(request, timeout):
            seen.append(request)
            return io.BytesIO(canonical(response()).encode())
        client = OpenRouterClient("secret-value", opener=opener)
        self.assertEqual(client.request("/chat/completions", {"model": "test/model"}), response())
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer secret-value")

    def test_errors_and_echoed_secrets_are_redacted(self):
        for exception in (HTTPError("url", 429, "secret-value", {}, None),
                          URLError("secret-value"), TimeoutError("secret-value")):
            def opener(request, timeout):
                raise exception
            client = OpenRouterClient("secret-value", opener=opener)
            with self.assertRaises(APIError) as caught:
                client.request("/chat/completions", {})
            self.assertNotIn("secret-value", str(caught.exception))
        client = OpenRouterClient("secret-value", opener=lambda *a, **kw:
                                  io.BytesIO(b'{"echo":"secret-value"}'))
        self.assertEqual(client.request("/models"), {"echo": "[REDACTED]"})
        client = OpenRouterClient("secret-value", opener=lambda *a, **kw:
                                  io.BytesIO(b'{"error":"secret-value"}'))
        with self.assertRaises(APIError):
            client.request("/models")

        def rejected(request, timeout):
            raise HTTPError("url", 404, "Not found", {}, io.BytesIO(
                b'{"error":{"message":"No endpoints secret-value","code":404}}'))
        with self.assertRaises(APIError) as caught:
            OpenRouterClient("secret-value", opener=rejected).request("/chat/completions", {})
        self.assertEqual(caught.exception.details["error"]["code"], 404)
        self.assertNotIn("secret-value", canonical(caught.exception.details))


class FakeEpisode:
    def __init__(self, **kwargs):
        self.rollout_count = kwargs["rollout_count"]
        self.output = Path(kwargs["output"])
        self.output.mkdir(parents=True)
        self.road, self.records, self.summary = kwargs["road"], [], None
        self.done = False

    def __enter__(self):
        return self

    def prompt(self):
        return self.road.prompt_features({"speed": 0})

    def apply(self, completion, completion_ids, prompt_ids):
        self.road.parse_completion(completion)
        self.records.append(completion)
        self.done = len(self.records) == 2

    def finish(self):
        self.summary = {"success": False, "reason": "stalled",
                        "reward_components": {"total": 0, "legal_progress_m": 0}}

    def __exit__(self, exc_type, exc, traceback):
        if exc_type:
            self.summary = {"reason": "infrastructure_failure", "training_eligible": False}
        else:
            self.finish()


class EpisodeTests(unittest.TestCase):
    def test_provider_pinned_after_first_call_and_results_saved(self):
        bodies = []
        def opener(request, timeout):
            bodies.append(strict_json(request.data.decode()))
            return io.BytesIO(canonical(response()).encode())
        client = OpenRouterClient("test-secret", opener=opener)
        config = configuration("test/model", ENTRY, seconds=30, temperature=.6, seed=1073, max_tokens=512)
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(client, model="test/model", config=config, godot="unused",
                                 output=Path(directory) / "episode", rollout=1, episode_factory=FakeEpisode)
            self.assertIsNone(result["infrastructure_error"])
            self.assertNotIn("only", bodies[0]["provider"])
            self.assertEqual(bodies[1]["provider"]["only"], ["Test Provider"])
            self.assertEqual(result["api_requests"], 2)
            self.assertFalse(result["training_eligible"])
            self.assertEqual(aggregate([result])["stalls"], 1)
            self.assertTrue((Path(directory) / "episode/benchmark.json").is_file())

    def test_provider_change_is_infrastructure_failure(self):
        calls = []
        def opener(request, timeout):
            calls.append(1)
            return io.BytesIO(canonical(response(provider="first" if len(calls) == 1 else "second")).encode())
        config = configuration("test/model", ENTRY, seconds=30, temperature=.6, seed=1073, max_tokens=512)
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(OpenRouterClient("test-secret", opener=opener), model="test/model",
                                 config=config, godot="unused", output=Path(directory) / "episode",
                                 rollout=1, episode_factory=FakeEpisode)
            self.assertIn("provider changed", result["infrastructure_error"])
            self.assertFalse(result["summary"]["training_eligible"])
            self.assertEqual(aggregate([result])["scored_attempts"], 0)

    def test_selected_provider_slug_and_declared_rollout_count(self):
        bodies = []
        def opener(request, timeout):
            bodies.append(strict_json(request.data.decode()))
            return io.BytesIO(canonical(response()).encode())
        config = configuration("test/model", ENTRY, seconds=30, temperature=.6, seed=1073, max_tokens=512)
        config["provider"]["only"] = ["test-provider/bf16"]
        config["selected_endpoint"] = {"provider_name": "Test Provider"}
        class CheckedEpisode(FakeEpisode):
            def __enter__(self):
                if self.rollout_count != 12:
                    raise ValueError("Wrong configured rollout count")
                return super().__enter__()
        with tempfile.TemporaryDirectory() as directory:
            result = run_episode(OpenRouterClient("test-secret", opener=opener), model="test/model",
                config=config, godot="unused", output=Path(directory) / "episode", rollout=1,
                rollout_count=12, episode_factory=CheckedEpisode)
            self.assertIsNone(result["infrastructure_error"])
            self.assertTrue(all(row["provider"]["only"] == ["test-provider/bf16"] for row in bodies))


@unittest.skipUnless(os.environ.get("THUNDERHILL_GODOT"), "Set THUNDERHILL_GODOT for real simulator test")
class NativeGodotTests(unittest.TestCase):
    def test_invalid_message_is_scored_and_http_failure_is_not(self):
        config = configuration("test/model", ENTRY, seconds=1, temperature=.6, seed=1073, max_tokens=512)
        directory = Path(tempfile.mkdtemp(prefix="openrouter-failures-native-",
                         dir=Path(__file__).resolve().parents[1] / "artifacts"))
        malformed = {"role": "assistant", "content": "I would accelerate."}
        client = OpenRouterClient("unused-secret", opener=lambda *a, **kw:
                                  io.BytesIO(canonical(response(malformed)).encode()))
        result = run_episode(client, model="test/model", config=config,
                             godot=os.environ["THUNDERHILL_GODOT"], output=directory / "invalid", rollout=1)
        self.assertIsNone(result["infrastructure_error"])
        self.assertEqual(result["summary"]["reason"], "invalid_model_action")
        self.assertEqual(result["summary"]["reward_components"]["syntax_penalty"], -1)
        self.assertEqual(result["summary"]["actions"], 0)
        self.assertTrue(result["summary"]["recording_provenance_verified"])

        def fail(*args, **kwargs):
            raise HTTPError("url", 503, "unused-secret", {}, None)
        result = run_episode(OpenRouterClient("unused-secret", opener=fail), model="test/model", config=config,
                             godot=os.environ["THUNDERHILL_GODOT"], output=directory / "http", rollout=1)
        self.assertIn("HTTP status 503", result["infrastructure_error"])
        self.assertNotIn("unused-secret", canonical(result))
        self.assertEqual(result["summary"]["reason"], "infrastructure_failure")
        self.assertNotIn("reward_components", result["summary"])
        self.assertFalse(result["summary"]["training_eligible"])
        self.assertTrue((directory / "http" / result["summary"]["video_job"]).is_file())

    def test_remote_idle_stops_and_archives_without_training(self):
        client = OpenRouterClient("unused-secret", opener=lambda *a, **kw:
                                  io.BytesIO(canonical(response()).encode()))
        config = configuration("test/model", ENTRY, seconds=30, temperature=.6, seed=1073, max_tokens=512)
        directory = Path(tempfile.mkdtemp(prefix="openrouter-native-",
                         dir=Path(__file__).resolve().parents[1] / "artifacts"))
        result = run_episode(client, model="test/model", config=config,
                             godot=os.environ["THUNDERHILL_GODOT"], output=directory / "episode", rollout=1)
        self.assertIsNone(result["infrastructure_error"], result["infrastructure_error"])
        summary = result["summary"]
        self.assertEqual(summary["reason"], "stalled")
        self.assertEqual(summary["actions"], 100)
        self.assertEqual(summary["recorded_transitions"], 1200)
        self.assertFalse(summary["training_eligible"])
        self.assertTrue(summary["recording_provenance_verified"])
        self.assertEqual(summary["reward_components"]["failure_penalty"], 0)
        self.assertTrue((directory / "episode" / summary["video_job"]).is_file())


if __name__ == "__main__":
    unittest.main()
