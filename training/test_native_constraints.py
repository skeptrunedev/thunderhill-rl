"""Actual pinned Gemma tokenizer and XGrammar, with CPU policy gradients."""
import unittest

import torch
import xgrammar as xgr
from transformers import AutoTokenizer

from model_runtime import GEMMA4_SPEC
from native_constraints import NativeToolConstraint
from native_tools import NativeBikeTools


class NativeConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            GEMMA4_SPEC.model, revision=GEMMA4_SPEC.revision, local_files_only=True,
        )
        cls.tools = NativeBikeTools(cls.tokenizer)
        cls.constraint = NativeToolConstraint(cls.tokenizer, len(cls.tokenizer))

    def completion(self, steer=0, throttle=0, front=0, rear=0):
        return ("<|tool_call>call:control_bike{front_brake_percent:" + str(front)
                + ",rear_brake_percent:" + str(rear) + ",steer_milli:" + str(steer)
                + ",throttle_percent:" + str(throttle) + "}<tool_call|><|tool_response>")

    def accepts(self, text):
        matcher = xgr.GrammarMatcher(self.constraint.compiled)
        for token in self.tokenizer.encode(text, add_special_tokens=False):
            if not matcher.accept_token(token):
                return False
        return matcher.is_terminated()

    def test_all_permitted_integers_and_native_template(self):
        for value in range(-1000, 1001):
            self.assertTrue(self.accepts(self.completion(steer=value)))
        for value in range(101):
            self.assertTrue(self.accepts(self.completion(throttle=value, front=value, rear=value)))
        native = self.tools.completion_from_controls(
            dict(steer=-1, throttle=1, front_brake=1, rear_brake=0),
        )
        self.assertTrue(self.accepts(native))
        self.assertLessEqual(len(self.tokenizer.encode(native, add_special_tokens=False)),
                             self.tools.max_completion_length)
        # Sampling can choose single character tokens instead of the usual
        # subwords. The rollout limit must cover this longest legal encoding.
        body = self.completion(steer=-1000, throttle=100, front=100, rear=100)
        body = body.removeprefix("<|tool_call>").removesuffix("<tool_call|><|tool_response>")
        character_ids = [48] + [token for char in body
                                for token in self.tokenizer.encode(char, add_special_tokens=False)] + [49, 50]
        matcher = xgr.GrammarMatcher(self.constraint.compiled)
        self.assertTrue(all(matcher.accept_token(token) for token in character_ids))
        self.assertTrue(matcher.is_terminated())
        self.assertLessEqual(len(character_ids), self.tools.max_completion_length)

    def test_rejects_bad_envelopes_ranges_and_multiple_calls(self):
        valid = self.completion()
        invalid = [self.completion(steer=-1001), self.completion(steer=1001),
                   self.completion(throttle=101), self.completion(front=-1),
                   self.completion(rear=101), self.completion(throttle="1.5"),
                   self.completion(steer="-0"), self.completion(throttle="01"),
                   valid + valid, valid.replace("control_bike", "other"),
                   valid.replace("<|tool_response>", "[eot]"),
                   valid.replace(",rear_brake_percent:0", ""),
                   valid.replace(",rear_brake_percent:0", ",front_brake_percent:0"),
                   valid.replace("<|tool_call>", ""), valid + "hello"]
        for text in invalid:
            with self.subTest(text=text):
                self.assertFalse(self.accepts(text))

    def test_masked_sampling_likelihoods_match_differentiable_training(self):
        # Both variable length rows include reserved tool handoff. No Gemma
        # weights are needed to prove matching probability distributions.
        texts = [self.completion(steer=-1000, throttle=100), self.completion(steer=5)]
        token_rows = [self.tokenizer.encode(text, add_special_tokens=False) for text in texts]
        width = max(map(len, token_rows))
        ids = torch.tensor([row + [0] * (width - len(row)) for row in token_rows])
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in token_rows])
        torch.manual_seed(9)
        # An actual trainable parameter tests gradient flow through gather,
        # normalization and policy likelihood, including later tool tokens.
        parameter = torch.nn.Parameter(torch.randn(len(self.tokenizer)) * 0.05)
        logits = parameter[None, None, :].expand(2, width, -1)
        processor = self.constraint.logits_processor()
        observed = []
        for index in range(width):
            prefix = torch.cat((torch.ones(2, 1, dtype=torch.long), ids[:, :index]), dim=1)
            scores = processor(prefix, logits[:, index].detach().clone())
            observed.append(scores.log_softmax(-1).gather(1, ids[:, index, None]).squeeze(1))
        observed = torch.stack(observed, dim=1).masked_fill(~mask.bool(), 0)
        logps, entropy = self.constraint.log_probs(logits, ids, mask, compute_entropy=True)
        torch.testing.assert_close(logps, observed, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((entropy >= -1e-6).all())
        self.assertEqual(logps[:, 0].tolist(), [0, 0])
        self.assertEqual(logps[0, -1].item(), 0)
        # Opposing trajectory advantages must train real integer choices.
        loss = -(logps * torch.tensor([[1.0], [-1.0]]) * mask).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(parameter.grad[48].item(), 0)
        self.assertEqual(parameter.grad[50].item(), 0)
        self.assertEqual(parameter.grad[0].item(), 0)
        bad_ids = ids.clone()
        bad_ids[0, -1] = 106
        with self.assertRaisesRegex(ValueError, "handoff"):
            self.constraint.log_probs(logits, bad_ids, mask)
        bad_ids = ids.clone()
        bad_ids[0, 0] = 49
        with self.assertRaisesRegex(ValueError, "violates"):
            self.constraint.log_probs(logits, bad_ids, mask)


if __name__ == "__main__":
    unittest.main()
