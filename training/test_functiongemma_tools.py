"""Actual pinned FunctionGemma tokenizer, template and differentiable grammar."""
import unittest
from unittest.mock import Mock, patch

from transformers import AutoTokenizer

from functiongemma_tools import FunctionGemmaBikeTools
from model_runtime import FUNCTIONGEMMA_SPEC, PolicyRoadTelemetry, load_base
from native_constraints import NativeToolConstraint
import test_native_constraints


class FunctionGemmaConstraintTests(test_native_constraints.NativeConstraintTests):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            FUNCTIONGEMMA_SPEC.model, revision=FUNCTIONGEMMA_SPEC.revision,
            local_files_only=True,
        )
        cls.tools = FunctionGemmaBikeTools(cls.tokenizer)
        cls.constraint = NativeToolConstraint(
            cls.tokenizer, len(cls.tokenizer), native_tools=cls.tools,
        )

    def completion(self, steer=0, throttle=0, front=0, rear=0):
        return ("<start_function_call>call:control_bike{front_brake_percent:" + str(front)
                + ",rear_brake_percent:" + str(rear) + ",steer_milli:" + str(steer)
                + ",throttle_percent:" + str(throttle)
                + "}<end_function_call><start_function_response>")

    def test_all_permitted_integers_and_native_template(self):
        # The superclass additionally tests a Gemma4 literal envelope; the
        # numeric acceptance and longest character encoding remain identical.
        for steer in range(-1000, 1001):
            self.assertTrue(self.accepts(self.completion(steer=steer)))
        for pedal in range(101):
            self.assertTrue(self.accepts(self.completion(throttle=pedal, front=pedal, rear=pedal)))
        controls = dict(steer=-1, throttle=1, front_brake=1, rear_brake=1)
        native = self.tools.completion_from_controls(controls)
        self.assertTrue(self.accepts(native))
        body = native.removeprefix(self.tools.tool_start).removesuffix(self.tools.tool_end + self.tools.tool_stop)
        ids = [48] + [token for char in body for token in self.tokenizer.encode(char, add_special_tokens=False)] + [49, 50]
        import xgrammar as xgr
        matcher = xgr.GrammarMatcher(self.constraint.compiled)
        self.assertTrue(all(matcher.accept_token(token) for token in ids))
        self.assertTrue(matcher.is_terminated())
        self.assertLessEqual(len(ids), self.tools.max_completion_length)

    def test_rejects_bad_envelopes_ranges_and_multiple_calls(self):
        for text in [self.completion(steer=-1001), self.completion(steer=1001),
                     self.completion(throttle=101), self.completion(front=-1),
                     self.completion(rear=101), self.completion(throttle="1.5"),
                     self.completion(steer="-0"), self.completion(throttle="01"),
                     self.completion() * 2, self.completion() + "hello",
                     self.completion().replace("control_bike", "other"),
                     self.completion().replace(self.tools.tool_stop, "<end_of_turn>"),
                     self.completion().replace(",rear_brake_percent:0", "")]:
            with self.subTest(text=text):
                self.assertFalse(self.accepts(text))

    def test_template_and_real_tool_feedback(self):
        tools = self.tools
        before, after = {"speed": 8}, {"speed": 9}
        controls = dict(steer=-0.125, throttle=0.65, front_brake=0.1, rear_brake=0, shift=0)
        completion = tools.completion_from_controls(controls)
        self.assertEqual(completion, self.completion(-125, 65, 10, 0))
        self.assertEqual(tools.parse_completion(completion), controls)
        prompt = tools.prompt(before)
        self.assertIn("You are a model that can do function calling with the following functions", prompt)
        self.assertIn("<start_function_declaration>declaration:control_bike", prompt)
        self.assertTrue(prompt.endswith("<start_of_turn>model\n"))
        feedback = tools.prompt(after, previous_completion=completion,
                                previous_features=before,
                                tool_response={"road": after, "tick": 12, "done": False})
        self.assertIn(completion + "response:control_bike{done:false,road:{speed:9},tick:12}<end_function_response>", feedback)
        self.assertTrue(feedback.endswith("<end_function_response>"))
        self.assertEqual(feedback.count("<start_function_response>"), 1)
        self.assertEqual(feedback.count("<start_function_call>"), 1)
        self.assertNotIn("<|tool_response>", feedback)
        ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        self.assertEqual(ids[-1], 50)
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=False), completion)
        for invalid in [completion * 2, completion.replace("65", "101"),
                        completion.replace("65", "65.0"),
                        completion.removesuffix(tools.tool_stop),
                        completion.replace("rear_brake_percent:0", "throttle_percent:0")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tools.parse_completion(invalid)
        road = PolicyRoadTelemetry(FUNCTIONGEMMA_SPEC, self.tokenizer)
        self.assertEqual(self.tokenizer.eos_token, tools.tool_stop)
        self.assertEqual(road.parse_completion(completion), controls)
        self.assertEqual(road.prompt_features(before), prompt)

    def test_fp32_pinned_loader(self):
        import torch
        with patch("transformers.AutoModelForCausalLM.from_pretrained") as loader:
            model = Mock()
            loader.return_value = model, {}
            self.assertIs(load_base(FUNCTIONGEMMA_SPEC, device="cpu"), model)
            loader.assert_called_once_with(
                FUNCTIONGEMMA_SPEC.model, revision=FUNCTIONGEMMA_SPEC.revision,
                dtype=torch.float32, attn_implementation="sdpa",
                device_map={"": "cpu"}, output_loading_info=True,
            )


if __name__ == "__main__":
    unittest.main()
