"""Native protocol tests against the actual pinned Gemma tokenizer, CPU only."""
import unittest

from transformers import AutoTokenizer

from model_runtime import GEMMA4_SPEC
from native_tools import NativeBikeTools


class NativeToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            GEMMA4_SPEC.model, revision=GEMMA4_SPEC.revision, local_files_only=True,
        )
        cls.codec = NativeBikeTools(cls.tokenizer)
        cls.controls = {"steer": -0.125, "throttle": 0.65, "front_brake": 0.1, "rear_brake": 0, "shift": 0}

    def test_native_schema_prompt(self):
        prompt = self.codec.prompt({"speed": 8.0, "curves": [0, 0.01]})
        self.assertIn("<|tool>declaration:control_bike{", prompt)
        self.assertIn('type:<|"|>INTEGER<|"|>', prompt)
        self.assertIn("integer from -1000 to 1000", prompt)
        self.assertIn('"speed":8.0', prompt)
        self.assertNotIn("STEER_MILLI", prompt)
        self.assertNotIn("<|think|>", prompt)
        self.assertTrue(prompt.endswith("<|turn>model\n"))
        self.assertEqual(prompt.count("<bos>"), 1)
        self.assertEqual(self.codec.stop_token_id, 50)
        with self.assertRaises(ValueError):
            self.codec.prompt({"speed": float("nan")})

    def test_template_roundtrip_and_response(self):
        completion = self.codec.completion_from_controls(self.controls)
        self.assertEqual(completion, "<|tool_call>call:control_bike{front_brake_percent:10,rear_brake_percent:0,steer_milli:-125,throttle_percent:65}<tool_call|><|tool_response>")
        self.assertEqual(self.codec.parse_completion(completion), self.controls)
        ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        self.assertEqual(ids[-1], self.codec.stop_token_id)
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=False), completion)
        self.assertNotEqual(self.tokenizer.decode(ids, skip_special_tokens=True), completion)
        history = self.codec._render(
            self.codec.messages({}) + [self.codec.assistant_message(self.controls, {"speed": 8, "tick": 12})],
            add_generation_prompt=True,
        )
        self.assertIn(completion + "response:control_bike{speed:8,tick:12}<tool_response|>", history)
        self.assertFalse(history.endswith("<|turn>model\n"))

    def test_bounded_actual_tool_feedback(self):
        before, after = {"speed": 7}, {"speed": 8}
        completion = self.codec.completion_from_controls(self.controls)
        prompt = self.codec.prompt(after, previous_completion=completion,
                                  previous_features=before,
                                  tool_response={"road": after, "tick": 12, "done": False})
        self.assertIn('"speed":7', prompt)
        self.assertIn("response:control_bike{done:false,road:{speed:8},tick:12}<tool_response|>", prompt)
        self.assertEqual(prompt.count("<|tool_call>"), 1)
        self.assertEqual(prompt.count("<|tool_response>"), 1)
        self.assertTrue(prompt.endswith("<tool_response|>"))
        with self.assertRaises(ValueError):
            self.codec.prompt(after, previous_completion=completion, previous_features=before,
                              tool_response={"road": before})
        with self.assertRaises(ValueError):
            self.codec.prompt(after, previous_completion=completion)

    def test_bounds_and_token_budget(self):
        for steer in (-1, 0, 1):
            for pedal in (0, 1):
                controls = {"steer": steer, "throttle": pedal, "front_brake": pedal, "rear_brake": pedal, "shift": 0}
                completion = self.codec.completion_from_controls(controls)
                self.assertEqual(self.codec.parse_completion(completion), controls)
                self.assertLess(len(self.tokenizer(completion, add_special_tokens=False)["input_ids"]), self.codec.max_completion_length)
        for field, value in (("steer", 1.1), ("throttle", -0.1), ("front_brake", True), ("rear_brake", float("inf")), ("shift", 1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.codec.completion_from_controls({**self.controls, field: value})

    def test_reject_malformed_or_multiple_calls(self):
        good = self.codec.completion_from_controls(self.controls)
        cases = [
            good + good, good[:-len("<|tool_response>")], good + "<turn|>",
            "Here is my call: " + good, good.replace("control_bike", "reset"),
            good.replace("throttle_percent:65", "throttle_percent:65.0"),
            good.replace("throttle_percent:65", "throttle_percent:true"),
            good.replace("throttle_percent:65", "throttle_percent:-1"),
            good.replace("throttle_percent:65", "throttle_percent:101"),
            good.replace("steer_milli:-125", "steer_milli:-1001"),
            good.replace("steer_milli:-125", "steer_milli:1001"),
            good.replace("throttle_percent:65", "throttle_percent:65,throttle_percent:65"),
            good.replace("throttle_percent:65", "unknown:65"),
            good.replace(",throttle_percent:65", ""),
            good.replace("throttle_percent:65", "throttle_percent:__import__('os')"),
            "control_bike 0 65 0 0",
        ]
        for malformed in cases:
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                self.codec.parse_completion(malformed)

    def test_argument_order_is_not_semantic(self):
        reordered = "<|tool_call>call:control_bike{throttle_percent:65,steer_milli:-125,rear_brake_percent:0,front_brake_percent:10}<tool_call|><|tool_response>"
        self.assertEqual(self.codec.parse_completion(reordered), self.controls)


if __name__ == "__main__":
    unittest.main()
