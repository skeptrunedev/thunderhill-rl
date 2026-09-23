"""Qwen3.8's official XML tool protocol for simulator actions.

Template source: Qwen/Qwen3.8-27B at QWEN_REVISION. This is a native
function call codec, not a text action imitation format.
"""
import json
import re

from native_tools import ARGUMENTS, NativeBikeTools

QWEN_MODEL = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class QwenBikeTools(NativeBikeTools):
    tool_start = "<tool_call>"
    tool_end = "</tool_call>"
    tool_stop = "<|im_end|>"
    action_version = "qwen38-native-bike-tools-v1"
    # Includes the longest legal character-by-character tokenization of XML.
    max_completion_length = 384

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        for attr, text in (("start_token_id", self.tool_start),
                           ("end_token_id", self.tool_end),
                           ("stop_token_id", self.tool_stop)):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != text:
                raise ValueError(f"Qwen native delimiter is not a reserved token: {text}")
            setattr(self, attr, ids[0])
        descriptions = {
            "steer_milli": "Steering input, integer from -1000 to 1000. Balance assistance is enabled.",
            "throttle_percent": "Throttle, integer from 0 to 100 percent.",
            "front_brake_percent": "Front brake, integer from 0 to 100 percent.",
            "rear_brake_percent": "Rear brake, integer from 0 to 100 percent.",
        }
        self.schema = {
            "type": "function", "function": {
                "name": "control_bike",
                "description": "Apply rider controls for 0.1 simulated seconds and return updated telemetry. Automatic gears are enabled.",
                "parameters": {"type": "object", "properties": {
                    name: {"type": "integer", "description": descriptions[name],
                           "minimum": bounds[2], "maximum": bounds[3]}
                    for name, bounds in ARGUMENTS.items()},
                    "required": list(ARGUMENTS), "additionalProperties": False},
            },
        }
        self.tools = [self.schema]
        # The official template emits arguments in dictionary insertion order.
        pieces = [f'Token({self.start_token_id})', json.dumps("\n<function=control_bike>\n")]
        for name in ARGUMENTS:
            pieces.extend([json.dumps(f"<parameter={name}>\n"),
                           "steer" if name == "steer_milli" else "pedal",
                           json.dumps("\n</parameter>\n")])
        pieces.extend([json.dumps("</function>\n"), f'Token({self.end_token_id})'])
        self.grammar = ('root ::= ' + ' '.join(pieces) + '\n'
                        'pedal ::= "0" | [1-9] [0-9]? | "100"\n'
                        'steer ::= "0" | "-"? ([1-9] [0-9]? [0-9]? | "1000")')

    @classmethod
    def parse_completion(cls, raw):
        if not isinstance(raw, str):
            raise ValueError("Qwen tool completion must be text")
        match = re.fullmatch(
            r"\s*<tool_call>\s*<function=control_bike>\s*(.*?)\s*</function>\s*</tool_call><\|im_end\|>\s*",
            raw, flags=re.DOTALL,
        )
        if match is None:
            raise ValueError("Expected exactly one complete native Qwen control_bike call")
        remaining = match[1]
        arguments = {}
        while remaining:
            item = re.match(r"<parameter=([a-z_]+)>\s*(-?(?:0|[1-9][0-9]*))\s*</parameter>\s*", remaining)
            if item is None:
                raise ValueError("Invalid native Qwen parameter")
            name, text = item[1], item[2]
            if name not in ARGUMENTS or name in arguments or text == "-0":
                raise ValueError("Duplicate, unknown, or noncanonical native Qwen parameter")
            value = int(text)
            if not ARGUMENTS[name][2] <= value <= ARGUMENTS[name][3]:
                raise ValueError("Native Qwen parameter outside allowed range")
            arguments[name] = value
            remaining = remaining[item.end():]
        if set(arguments) != set(ARGUMENTS):
            raise ValueError("Native Qwen call requires all four arguments")
        return {**{physical: arguments[name] / scale
                   for name, (physical, scale, _, _) in ARGUMENTS.items()}, "shift": 0}

    def assistant_message(self, controls, response=None):
        if response is not None:
            raise ValueError("Qwen tool results require a separate role tool message")
        return {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {
                "name": "control_bike", "arguments": self._arguments(controls)}}]}

    def feedback_messages(self, controls, response):
        return [self.assistant_message(controls),
                {"role": "tool", "name": "control_bike",
                 "content": json.dumps(response, separators=(",", ":"), allow_nan=False)}]

    def completion_from_controls(self, controls):
        # Derive native XML from the pinned template, removing only the
        # post-turn separator newline that generation stops before producing.
        return super().completion_from_controls(controls).rstrip("\n")
