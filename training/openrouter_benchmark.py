"""Compare untouched hosted policies through native tools and actual gameplay.

This is evaluation only. Hosted request fingerprints identify configuration,
not inaccessible model weights. No demonstrations or optimizer updates occur.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lap_episode import DEFAULT_STALL_CONFIG, LapEpisode, REWARD_VERSION
from lap_policy import OBSERVATION_VERSION, RoadTelemetry
from native_tools import ARGUMENTS, NativeBikeTools

API = "https://openrouter.ai/api/v1"
IDENTITY_KIND = "api_request_config_sha256"


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class APINativeTools:
    action_version = "openrouter-native-bike-tools-v1"

    def __init__(self):
        descriptions = {
            "steer_milli": "Steering input, integer from -1000 to 1000. Balance assistance is enabled.",
            "throttle_percent": "Throttle, integer from 0 to 100 percent.",
            "front_brake_percent": "Front brake, integer from 0 to 100 percent.",
            "rear_brake_percent": "Rear brake, integer from 0 to 100 percent.",
        }
        self.tools = [{"type": "function", "function": {
            "name": "control_bike",
            "description": "Apply rider controls for 0.1 simulated seconds and return updated telemetry. Automatic gears are enabled.",
            "parameters": {"type": "object", "properties": {
                name: {"type": "integer", "description": descriptions[name],
                       "minimum": bounds[2], "maximum": bounds[3]}
                for name, bounds in ARGUMENTS.items()
            }, "required": list(ARGUMENTS), "additionalProperties": False},
        }}]

    @staticmethod
    def messages(features):
        return NativeBikeTools.messages(None, features)

    @staticmethod
    def parse_completion(completion):
        message = strict_json(completion)
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("Expected native assistant message")
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Expected exactly one native tool call")
        call = calls[0]
        if (not isinstance(call, dict) or call.get("type") != "function"
                or not isinstance(call.get("id"), str) or not call["id"]):
            raise ValueError("Missing native function identity")
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != "control_bike":
            raise ValueError("Expected control_bike function")
        if not isinstance(function.get("arguments"), str):
            raise ValueError("Expected JSON arguments string")
        arguments = strict_json(function["arguments"])
        if not isinstance(arguments, dict) or set(arguments) != set(ARGUMENTS):
            raise ValueError("Expected exactly four control arguments")
        controls = {"shift": 0}
        for name, (physical, scale, low, high) in ARGUMENTS.items():
            value = arguments[name]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid native control argument: {name}")
            controls[physical] = value / scale
        return controls

    def prompt(self, features, *, previous_completion=None,
               previous_features=None, tool_response=None):
        if previous_completion is None:
            if previous_features is not None or tool_response is not None:
                raise ValueError("Unexpected prior interaction")
            return canonical(self.messages(features))
        if previous_features is None or tool_response is None or tool_response.get("road") != features:
            raise ValueError("Tool feedback does not match observation")
        self.parse_completion(previous_completion)
        previous = strict_json(previous_completion)
        messages = self.messages(previous_features)
        messages.extend([previous, {"role": "tool",
            "tool_call_id": previous["tool_calls"][0]["id"],
            "content": canonical(tool_response)}])
        return canonical(messages)


class APIRoadTelemetry(RoadTelemetry):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.native_tools = APINativeTools()

    def prompt_features(self, features):
        return self.native_tools.prompt(features)

    def parse_completion(self, completion):
        return self.native_tools.parse_completion(completion)


class APIError(RuntimeError):
    """Infrastructure failure with a credential safe diagnostic."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


class OpenRouterClient:
    def __init__(self, key, *, timeout=90, opener=urlopen):
        if not key or any(c.isspace() for c in key):
            raise ValueError("Invalid OpenRouter credential")
        self._key, self.timeout, self._opener = key, timeout, opener

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, "[REDACTED]")
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {self.redact(key): self.redact(item) for key, item in value.items()}
        return value

    def request(self, path, body=None):
        request = Request(API + path, data=canonical(body).encode() if body is not None else None,
                          headers={"Authorization": "Bearer " + self._key,
                                   "Content-Type": "application/json"})
        try:
            with self._opener(request, timeout=self.timeout) as response:
                result = strict_json(response.read().decode())
        except HTTPError as error:
            # Retain sanitized routing diagnostics, never request headers.
            try:
                details = self.redact(strict_json(error.read(65536).decode()))
            except (ValueError, OSError, AttributeError):
                details = None
            raise APIError(f"OpenRouter HTTP status {error.code}", details=details) from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise APIError("OpenRouter transport or JSON response failure") from None
        result = self.redact(result)
        if not isinstance(result, dict) or "error" in result:
            raise APIError("OpenRouter returned an API error")
        return result


def validate_models(catalog, models):
    entries = {row["id"]: row for row in catalog.get("data", [])}
    selected = {}
    required = {"tools", "tool_choice", "temperature", "max_tokens"}
    for model in models:
        if model not in entries:
            raise ValueError(f"Model not present in OpenRouter catalog: {model}")
        entry = entries[model]
        missing = required - set(entry.get("supported_parameters", []))
        if missing:
            raise ValueError(f"Model {model} lacks required parameters: {sorted(missing)}")
        selected[model] = entry
    return selected


def select_endpoint(config, response):
    """Intersect capabilities at one endpoint, never the model catalog union.

    Forced function and required both demand our sole declared native tool.
    Prefer the model's highest available floating point precision, then a stable
    endpoint slug. Do not drop requested supported parameters to find a route.
    """
    needed = {"tools", "tool_choice", "temperature", "max_tokens"}
    if config["seed"] is not None:
        needed.add("seed")
    if config.get("reasoning") is not None:
        needed.add("reasoning")
    candidates = []
    for endpoint in response.get("data", {}).get("endpoints", []):
        if not needed.issubset(endpoint.get("supported_parameters", [])):
            continue
        if (not endpoint.get("tag") or not endpoint.get("provider_name")
                or (endpoint.get("max_completion_tokens") is not None
                    and endpoint["max_completion_tokens"] < config["max_tokens"])):
            continue
        capabilities = endpoint.get("supports_tool_choice", {})
        if capabilities.get("function") is True:
            choice = {"type": "function", "function": {"name": "control_bike"}}
        elif capabilities.get("required") is True:
            choice = "required"
        else:
            continue
        candidates.append((endpoint, choice))
    if not candidates:
        raise ValueError(f"No single endpoint for {config['model']} supports {sorted(needed)} and a forced native tool")
    precision_order = {"bf16": 0, "fp16": 1, "fp8": 2, "unknown": 3, "fp4": 4}
    endpoint, choice = min(candidates, key=lambda row: (
        precision_order.get(row[0].get("quantization"), 3), row[0]["tag"]))
    return {**config, "selected_endpoint": endpoint, "tool_choice": choice,
            "provider": {**config["provider"], "only": [endpoint["tag"]]}}


def configuration(model, catalog_entry, *, seconds, temperature, seed, max_tokens, reasoning=None):
    return {"model": model, "time_budget_seconds": seconds, "temperature": temperature,
            "seed": seed if "seed" in catalog_entry["supported_parameters"] else None,
            "requested_seed": seed, "max_tokens": max_tokens,
            "requested_reasoning": reasoning,
            "reasoning": ({"enabled": reasoning == "enabled"}
                          if reasoning is not None and "reasoning" in catalog_entry["supported_parameters"] else None),
            "observation_version": OBSERVATION_VERSION, "reward_version": REWARD_VERSION,
            "action_version": APINativeTools.action_version,
            "stall_config": asdict(DEFAULT_STALL_CONFIG), "policy_identity_kind": IDENTITY_KIND,
            "tools": APINativeTools().tools, "prompt_template": APINativeTools.messages({}),
            "provider": {"require_parameters": True, "allow_fallbacks": False},
            "tool_choice": "required", "training_method": "evaluation_only",
            "supervised_training_performed": False}


def run_episode(client, *, model, config, godot, output, rollout, rollout_count=None, provider=None,
                episode_factory=LapEpisode):
    digest = hashlib.sha256(canonical(config).encode()).hexdigest()
    road = APIRoadTelemetry()
    started = time.monotonic()
    episode = None
    requests = 0
    resolved_model = None
    failure = None
    failure_details = None
    expected_provider = config.get("selected_endpoint", {}).get("provider_name") or provider
    try:
        with episode_factory(godot=godot, output=output, road=road, adapter_sha256=digest,
                policy_identity_kind=IDENTITY_KIND, model=model, revision="hosted-unpinned",
                generation=0, rollout=rollout, rollout_count=rollout_count or rollout,
                evaluation=True, time_budget_seconds=config["time_budget_seconds"]) as episode:
            with (Path(output) / "api_responses.jsonl").open("w") as log:
                while not episode.done:
                    routing = dict(config["provider"])
                    if provider and "only" not in routing:
                        routing["only"] = [provider]
                    body = {"model": model, "messages": strict_json(episode.prompt()),
                            "tools": road.native_tools.tools, "tool_choice": config["tool_choice"],
                            "temperature": config["temperature"], "max_tokens": config["max_tokens"],
                            "provider": routing}
                    if config["seed"] is not None:
                        body["seed"] = config["seed"] + requests
                    if config.get("reasoning") is not None:
                        body["reasoning"] = config["reasoning"]
                    request_started = time.monotonic()
                    response = client.request("/chat/completions", body)
                    requests += 1
                    event = {"action_index": len(episode.records),
                             "latency_seconds": time.monotonic() - request_started,
                             "request": body, "response": response}
                    log.write(canonical(event) + "\n")
                    log.flush()
                    actual_provider, actual_model = response.get("provider"), response.get("model")
                    if not isinstance(actual_provider, str) or not actual_provider:
                        raise APIError("OpenRouter omitted provider identity")
                    if expected_provider is not None and expected_provider != actual_provider:
                        raise APIError("OpenRouter provider changed during benchmark")
                    provider = actual_provider
                    expected_provider = actual_provider
                    if not isinstance(actual_model, str) or not actual_model:
                        raise APIError("OpenRouter omitted resolved model identity")
                    if resolved_model is not None and resolved_model != actual_model:
                        raise APIError("OpenRouter resolved model changed during episode")
                    resolved_model = actual_model
                    choices = response.get("choices")
                    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                        raise APIError("OpenRouter returned invalid choices envelope")
                    message = choices[0].get("message")
                    # Invalid native messages are scored as invalid actions, never repaired.
                    episode.apply(canonical(message), [], [])
            episode.finish()
    except Exception as error:
        failure = client.redact(f"{type(error).__name__}: {error}")
        failure_details = client.redact(getattr(error, "details", None))
    summary = episode.summary if episode is not None else None
    result = {"model": model, "resolved_model": resolved_model, "provider": provider,
              "rollout": rollout, "policy_identity_kind": IDENTITY_KIND,
              "request_config_sha256": digest, "configuration": config,
              "summary": summary, "api_requests": requests,
              "wall_seconds": time.monotonic() - started, "infrastructure_error": failure,
              "infrastructure_error_details": failure_details,
              "optimizer_updates": 0, "training_eligible": False}
    Path(output).mkdir(parents=True, exist_ok=True)
    publish(Path(output) / "benchmark.json", result)
    return result


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def aggregate(results):
    scored = [r["summary"] for r in results if r["infrastructure_error"] is None
              and r["summary"] and "reward_components" in r["summary"]]
    return {"attempts": len(results), "scored_attempts": len(scored),
            "infrastructure_failures": sum(r["infrastructure_error"] is not None for r in results),
            "completed_laps": sum(row["success"] for row in scored),
            "stalls": sum(row["reason"] == "stalled" for row in scored),
            "invalid_calls": sum(row["reason"] == "invalid_model_action" for row in scored),
            "mean_reward": mean(row["reward_components"]["total"] for row in scored) if scored else None,
            "median_reward": median(row["reward_components"]["total"] for row in scored) if scored else None,
            "mean_progress_m": mean(row["reward_components"]["legal_progress_m"] for row in scored) if scored else None,
            "median_progress_m": median(row["reward_components"]["legal_progress_m"] for row in scored) if scored else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma separated exact OpenRouter model identifiers")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1073)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning", choices=("enabled", "disabled"),
                        help="Explicit reasoning setting where the model supports this parameter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--key-file", type=Path)
    args = parser.parse_args()
    models = [model.strip() for model in args.models.split(",")]
    if (not all(models) or len(set(models)) != len(models) or args.episodes < 1
            or not math.isfinite(args.seconds) or args.seconds <= 0
            or not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2
            or args.max_tokens < 1):
        parser.error("Invalid benchmark parameters")
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenRouterClient(key)
    catalog = validate_models(client.request("/models"), models)
    endpoint_catalog = {model: client.request("/models/" + model + "/endpoints") for model in models}
    selected_configs = {model: select_endpoint(configuration(model, catalog[model], seconds=args.seconds,
        temperature=args.temperature, seed=args.seed, max_tokens=args.max_tokens, reasoning=args.reasoning),
        endpoint_catalog[model]) for model in models}
    args.output.mkdir(parents=True, exist_ok=False)
    campaign = {"planned_models": models, "episodes_per_model": args.episodes,
                "catalog": catalog, "endpoint_catalog": endpoint_catalog,
                "selected_configurations": selected_configs, "results": [], "complete": False,
                "training_method": "evaluation_only", "optimizer_updates": 0,
                "supervised_training_performed": False}
    publish(args.output / "campaign.json", campaign)
    campaign_lock = threading.Lock()
    def run_model(model):
        provider = None
        results = []
        for index in range(args.episodes):
            config = {**selected_configs[model], "requested_seed": args.seed + index * 10000}
            if config["seed"] is not None:
                config["seed"] = config["requested_seed"]
            result = run_episode(client, model=model, config=config, godot=args.godot,
                                 output=args.output / model.replace("/", "__") / f"rollout-{index + 1:04d}",
                                 rollout=index + 1, rollout_count=args.episodes, provider=provider)
            provider = result["provider"] or provider
            results.append(result)
            with campaign_lock:
                campaign["results"].append(result)
                publish(args.output / "campaign.json", campaign)
            # A per model ledger persists every completed attempt independently.
            publish(args.output / (model.replace("/", "__") + ".json"),
                    {"model": model, "results": results, "aggregate": aggregate(results)})
            print(canonical({"model": model, "rollout": index + 1,
                             "aggregate": aggregate(results)}), flush=True)
            if result["infrastructure_error"]:
                break
        return results
    with ThreadPoolExecutor(max_workers=min(3, len(models))) as pool:
        futures = [pool.submit(run_model, model) for model in models]
        for future in as_completed(futures):
            future.result()
    campaign["complete"] = len(campaign["results"]) == len(models) * args.episodes
    campaign["aggregates"] = {model: aggregate([row for row in campaign["results"] if row["model"] == model])
                              for model in models}
    publish(args.output / "campaign.json", campaign)
    print(canonical(campaign["aggregates"]), flush=True)


if __name__ == "__main__":
    main()
