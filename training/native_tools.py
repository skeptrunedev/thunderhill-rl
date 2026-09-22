"""Gemma 4 native tool declarations, serialization and strict bike dispatch.

Protocol source: Google's Gemma 4 function calling guide and the pinned
Gemma 4 canonical chat template. No free text action codec is used here.
"""
from __future__ import annotations

import json
import math
import re

ARGUMENTS = {
    "steer_milli": ("steer", 1000, -1000, 1000),
    "throttle_percent": ("throttle", 100, 0, 100),
    "front_brake_percent": ("front_brake", 100, 0, 100),
    "rear_brake_percent": ("rear_brake", 100, 0, 100),
}
TOOL_START = "<|tool_call>"
TOOL_END = "<tool_call|>"
TOOL_STOP = "<|tool_response>"
ACTION_VERSION = "gemma4-native-bike-tools-v1"


class NativeBikeTools:
    """One native control_bike call, executed against the current observation.

    The simulator harness, not the model, attaches the current observation
    receipt. Generation must retain special tokens and stop on tool_response.
    """

    max_completion_length = 128

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        for token, expected in ((TOOL_START, 48), (TOOL_END, 49), (TOOL_STOP, 50)):
            if tokenizer.convert_tokens_to_ids(token) != expected:
                raise ValueError(f"Unexpected Gemma 4 native tool token: {token}")
        self.stop_token_id = tokenizer.convert_tokens_to_ids(TOOL_STOP)
        descriptions = {
            "steer_milli": "Steering input, integer from -1000 to 1000. Balance assistance is enabled.",
            "throttle_percent": "Throttle, integer from 0 to 100 percent.",
            "front_brake_percent": "Front brake, integer from 0 to 100 percent.",
            "rear_brake_percent": "Rear brake, integer from 0 to 100 percent.",
        }
        self.schema = {
            "type": "function",
            "function": {
                "name": "control_bike",
                "description": "Apply rider controls for 0.1 simulated seconds and return updated telemetry. Automatic gears are enabled.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {"type": "integer", "description": descriptions[name],
                               "minimum": bounds[2], "maximum": bounds[3]}
                        for name, bounds in ARGUMENTS.items()
                    },
                    "required": list(ARGUMENTS),
                    "additionalProperties": False,
                },
            },
        }
        self.tools = [self.schema]

    def messages(self, features: dict) -> list[dict]:
        # allow_nan=False prevents invalid telemetry from entering the prompt.
        telemetry = json.dumps(features, separators=(",", ":"), allow_nan=False)
        return [
            {"role": "system", "content":
             "Race forward and complete the lap as quickly as possible while staying on track. "
             "Call control_bike exactly once using integer arguments. Wait for updated telemetry before choosing another action."},
            {"role": "user", "content":
             "Simulator road guidance: speed m/s, lean rad, angle rad to lookahead center, "
             "distance m to it, lateral m, signed curves 1/m at 0,30,60,90,120m. "
             "Choose the next rider controls.\n" + telemetry},
        ]

    def _render(self, messages, *, add_generation_prompt):
        return self.tokenizer.apply_chat_template(
            messages, tools=self.tools, tokenize=False,
            add_generation_prompt=add_generation_prompt, enable_thinking=False,
        )

    def prompt(self, features: dict, *, previous_completion: str | None = None,
               previous_features: dict | None = None, tool_response: dict | None = None) -> str:
        """Render a fresh decision or the single latest tool interaction.

        A rolling pair bounds context size independently of lap duration. The
        tool response contains actual postaction simulator telemetry, never a
        model prediction. The previous native call is validated before reuse.
        """
        if previous_completion is None:
            if previous_features is not None or tool_response is not None:
                raise ValueError("Tool feedback requires a previous completion")
            return self._render(self.messages(features), add_generation_prompt=True)
        if previous_features is None or tool_response is None:
            raise ValueError("Previous native call requires its observation and tool result")
        if tool_response.get("road") != features:
            raise ValueError("Tool response does not match current road telemetry")
        json.dumps(tool_response, allow_nan=False)
        controls = self.parse_completion(previous_completion)
        messages = self.messages(previous_features)
        messages.append(self.assistant_message(controls, tool_response))
        return self._render(messages, add_generation_prompt=True)

    @staticmethod
    def parse_completion(raw: str) -> dict:
        """Validate the whole native envelope before producing physical controls."""
        if not isinstance(raw, str):
            raise ValueError("Native tool completion must be text")
        match = re.fullmatch(
            r"\s*<\|tool_call>call:control_bike\{([^{}]*)\}<tool_call\|><\|tool_response>\s*",
            raw,
        )
        if match is None:
            raise ValueError("Expected exactly one complete native control_bike tool call")
        arguments = {}
        for item in match[1].split(","):
            pair = re.fullmatch(r"\s*([a-z_]+)\s*:\s*(-?(?:0|[1-9][0-9]*))\s*", item)
            if pair is None or pair[1] not in ARGUMENTS or pair[1] in arguments:
                raise ValueError("Invalid, duplicate, or unknown native control argument")
            name, value = pair[1], int(pair[2])
            _, _, low, high = ARGUMENTS[name]
            if not low <= value <= high:
                raise ValueError(f"Native control {name} outside allowed range")
            arguments[name] = value
        if set(arguments) != set(ARGUMENTS):
            raise ValueError("Native control call must contain all four arguments")
        return {
            **{physical: arguments[name] / scale
               for name, (physical, scale, _, _) in ARGUMENTS.items()},
            "shift": 0,
        }

    @staticmethod
    def _arguments(controls: dict) -> dict:
        arguments = {}
        if controls.get("shift", 0) != 0:
            raise ValueError("Native controls use automatic gears")
        for name, (physical, scale, low, high) in ARGUMENTS.items():
            value = controls[physical]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Controls must be finite numbers")
            scaled = value * scale
            if not low <= scaled <= high:
                raise ValueError("Control outside allowed range")
            # Offline demonstration quantization matches the existing integer
            # action space. Generated calls themselves are never rounded.
            arguments[name] = round(scaled)
        return arguments

    def assistant_message(self, controls: dict, response: dict | None = None) -> dict:
        message = {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {
                "name": "control_bike", "arguments": self._arguments(controls),
            }}
        ]}
        if response is not None:
            message["tool_responses"] = [{"name": "control_bike", "response": response}]
        return message

    def completion_from_controls(self, controls: dict) -> str:
        # Derive the envelope from the pinned template, including the handoff
        # terminator. No assistant turn EOS is appended to a pending tool call.
        messages = self.messages({})
        prefix = self._render(messages, add_generation_prompt=True)
        full = self._render(messages + [self.assistant_message(controls)], add_generation_prompt=False)
        if not full.startswith(prefix):
            raise ValueError("Native chat template changed the generation prefix")
        completion = full[len(prefix):]
        self.parse_completion(completion)
        return completion
