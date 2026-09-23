"""Native Qwen template, dispatch, and constrained policy likelihood tests."""
import unittest

import torch
import xgrammar as xgr
from transformers import AutoTokenizer

from qwen_tools import QWEN_MODEL, QWEN_REVISION, QwenBikeTools
from native_constraints import NativeToolConstraint


class QwenToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, revision=QWEN_REVISION, local_files_only=True)
        cls.tools = QwenBikeTools(cls.tokenizer)
        cls.controls = dict(steer=-0.125, throttle=0.65, front_brake=0.1, rear_brake=0, shift=0)
        cls.constraint = NativeToolConstraint(cls.tokenizer, len(cls.tokenizer), native_tools=cls.tools)

    def accepts(self, text):
        matcher = xgr.GrammarMatcher(self.constraint.compiled)
        return all(matcher.accept_token(t) for t in self.tokenizer.encode(text, add_special_tokens=False)) and matcher.is_terminated()

    def test_template_roundtrip(self):
        prompt = self.tools.prompt({"speed": 7})
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertIn('"minimum": -1000', prompt)
        completion = self.tools.completion_from_controls(self.controls)
        self.assertTrue(completion.startswith('<tool_call>\n<function=control_bike>\n'))
        self.assertTrue(completion.endswith('</function>\n</tool_call><|im_end|>'))
        self.assertEqual(self.tools.parse_completion(completion), self.controls)
        self.assertTrue(self.accepts(completion))
        after = {"speed": 8}
        feedback = self.tools.prompt(after, previous_completion=completion, previous_features={"speed": 7},
                                     tool_response={"road": after, "tick": 12})
        self.assertIn('<|im_start|>user\n<tool_response>\n{"road":{"speed":8},"tick":12}\n</tool_response>', feedback)
        self.assertEqual(feedback.count('<function=control_bike>'), 1)
        with self.assertRaises(ValueError):
            self.tools.prompt(after, previous_completion=completion, previous_features={}, tool_response={"road": {}})

    def test_invalid_calls_fail_closed(self):
        good = self.tools.completion_from_controls(self.controls)
        for bad in [good + good, 'reasoning\n' + good, good + 'text', good.replace('control_bike', 'reset'),
                    good.replace('\n65\n', '\n65.0\n'), good.replace('\n65\n', '\ntrue\n'),
                    good.replace('\n65\n', '\n101\n'), good.replace('\n65\n', '\n-1\n'),
                    good.replace('\n-125\n', '\n-1001\n'), good.replace('\n-125\n', '\n-0\n'),
                    good.replace('throttle_percent', 'steer_milli'), good.replace('throttle_percent', 'unknown'),
                    good.replace('<|im_end|>', ''), good.replace('\n65\n', '\n065\n')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.tools.parse_completion(bad)
            self.assertFalse(self.accepts(bad))

    def test_bounds_and_character_encoding_budget(self):
        for steer in (-1, 0, 1):
            for pedal in (0, 1):
                controls = dict(steer=steer, throttle=pedal, front_brake=pedal, rear_brake=pedal, shift=0)
                text = self.tools.completion_from_controls(controls)
                self.assertEqual(self.tools.parse_completion(text), controls)
                self.assertTrue(self.accepts(text))
        text = self.tools.completion_from_controls(dict(steer=-1, throttle=1, front_brake=1, rear_brake=1))
        body = text.removeprefix(self.tools.tool_start).removesuffix(self.tools.tool_end + self.tools.tool_stop)
        ids = [self.tools.start_token_id] + [i for char in body for i in self.tokenizer.encode(char, add_special_tokens=False)] + [self.tools.end_token_id, self.tools.stop_token_id]
        matcher = xgr.GrammarMatcher(self.constraint.compiled)
        self.assertTrue(all(matcher.accept_token(i) for i in ids))
        self.assertTrue(matcher.is_terminated())
        self.assertLessEqual(len(ids), self.tools.max_completion_length)

    def test_sampling_and_training_likelihood_match(self):
        text = self.tools.completion_from_controls(self.controls)
        ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=False)])
        mask = torch.ones_like(ids)
        parameter = torch.nn.Parameter(torch.zeros(len(self.tokenizer)))
        logits = parameter[None, None, :].expand(1, ids.shape[1], -1)
        processor = self.constraint.logits_processor()
        observed = []
        for i in range(ids.shape[1]):
            prefix = torch.cat((torch.ones(1, 1, dtype=torch.long), ids[:, :i]), dim=1)
            scores = processor(prefix, logits[:, i].detach().clone())
            observed.append(scores.log_softmax(-1).gather(1, ids[:, i:i+1]).squeeze(1))
        expected = torch.stack(observed, dim=1)
        logps, entropy = self.constraint.log_probs(logits, ids, mask, compute_entropy=True)
        torch.testing.assert_close(logps, expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(entropy).all())
        (-logps.sum()).backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
