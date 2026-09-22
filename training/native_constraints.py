"""One native Gemma tool grammar shared by sampling and policy gradients.

XGrammar's token masks run outside the model forward and do not change its
SDPA, cache or CUDA graph behavior. Training gathers the allowed logits into
a small differentiable tensor instead of applying an inference-only in-place
kernel to tensors which require gradients.
"""
from collections import OrderedDict

import numpy as np
import torch
import xgrammar as xgr

from native_tools import ARGUMENTS, NativeBikeTools


class NativeToolConstraint:
    def __init__(self, tokenizer, vocab_size, *, native_tools=None):
        self.tools = native_tools if native_tools is not None else NativeBikeTools(tokenizer)
        self.vocab_size = vocab_size
        # Keep the explicit integer grammar coupled to the dispatch schema.
        bounds = {name: (values[2], values[3]) for name, values in ARGUMENTS.items()}
        if bounds != {"front_brake_percent": (0, 100), "rear_brake_percent": (0, 100),
                      "steer_milli": (-1000, 1000), "throttle_percent": (0, 100)}:
            raise ValueError("Native action schema changed; update its grammar")
        info = xgr.TokenizerInfo.from_huggingface(
            tokenizer, vocab_size=vocab_size, stop_token_ids=[self.tools.stop_token_id],
        )
        # The handoff token is the grammar's stop token, not part of its body.
        # Token macros require the real reserved IDs rather than lookalike text.
        grammar = '''root ::= Token(START_ID) "call:control_bike{front_brake_percent:" pedal ",rear_brake_percent:" pedal ",steer_milli:" steer ",throttle_percent:" pedal "}" Token(END_ID)
pedal ::= "0" | [1-9] [0-9]? | "100"
steer ::= "0" | "-"? ([1-9] [0-9]? [0-9]? | "1000")'''
        grammar = grammar.replace("START_ID", str(self.tools.start_token_id)).replace("END_ID", str(self.tools.end_token_id))
        self.compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
        # Bound retained prefixes for long campaigns. Each entry stores only
        # permitted IDs, not a dense vocabulary-sized mask.
        self._allowed_cache = OrderedDict()

    def logits_processor(self):
        """Create fresh matcher state for each independent generate() call."""
        return xgr.contrib.hf.LogitsProcessor(self.compiled)

    def _allowed(self, prefix, matcher):
        if prefix in self._allowed_cache:
            self._allowed_cache.move_to_end(prefix)
            return self._allowed_cache[prefix]
        if matcher.is_terminated():
            raise ValueError("Completion contains tokens after native tool handoff")
        bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
        matcher.fill_next_token_bitmask(bitmask)
        # XGrammar stores token i in bit i%32, word i//32. Explicit little
        # endian conversion also works on hosts whose native endian differs.
        packed = bitmask.numpy().astype("<i4", copy=False).view(np.uint8)
        allowed = np.flatnonzero(np.unpackbits(packed, bitorder="little")[:self.vocab_size])
        if not len(allowed):
            raise ValueError("Native grammar has no legal next token")
        result = torch.from_numpy(allowed.astype(np.int64))
        self._allowed_cache[prefix] = result
        if len(self._allowed_cache) > 8192:
            self._allowed_cache.popitem(last=False)
        return result

    def log_probs(self, logits, completion_ids, completion_mask, *, compute_entropy=False):
        """Differentiable likelihood under exactly the rollout grammar.

        Logits have shape [batch, completion, vocabulary] and must already be
        divided by sampling temperature. Padding does not participate in the
        grammar or gradients. Every unpadded row must include the handoff token.
        """
        if logits.ndim != 3 or logits.shape[:2] != completion_ids.shape:
            raise ValueError("Native likelihood tensor shapes differ")
        if logits.shape[-1] != self.vocab_size or completion_mask.shape != completion_ids.shape:
            raise ValueError("Native likelihood vocabulary or mask shape differs")
        ids = completion_ids.detach().cpu().tolist()
        masks = completion_mask.detach().cpu().tolist()
        allowed_rows = []
        for row, mask in zip(ids, masks, strict=True):
            count = sum(mask)
            if not count or mask != [1] * count + [0] * (len(mask) - count):
                raise ValueError("Native completion mask must contain a contiguous token prefix")
            if row[count - 1] != self.tools.stop_token_id:
                raise ValueError("Native completion is missing its tool handoff")
            prefix = ()
            matcher = xgr.GrammarMatcher(self.compiled)
            for index, token in enumerate(row):
                if index >= count:
                    allowed_rows.append(torch.tensor([0], dtype=torch.long))
                    continue
                allowed = self._allowed(prefix, matcher)
                if not matcher.accept_token(token):
                    raise ValueError("Sampled token violates native tool grammar")
                allowed_rows.append(allowed)
                prefix += (token,)
        lengths = torch.tensor([len(row) for row in allowed_rows], device=logits.device)
        indices = torch.nn.utils.rnn.pad_sequence(allowed_rows, batch_first=True).to(logits.device)
        legal = torch.arange(indices.shape[1], device=logits.device)[None, :] < lengths[:, None]
        flat = logits.reshape(-1, self.vocab_size)
        allowed_logits = flat.gather(1, indices).float().masked_fill(~legal, -torch.inf)
        # Fail closed on NaN/+inf on legal choices; -inf on some choices is
        # allowed, but at least one choice must retain finite probability.
        if torch.isnan(allowed_logits).any() or torch.isposinf(allowed_logits).any():
            raise FloatingPointError("Nonfinite native policy logits")
        normalizer = torch.logsumexp(allowed_logits, dim=-1)
        if not torch.isfinite(normalizer).all():
            raise FloatingPointError("No finite native policy choices")
        active = completion_mask.to(device=logits.device, dtype=torch.bool)
        selected_ids = completion_ids.to(logits.device).masked_fill(~active, 0)
        selected = flat.gather(1, selected_ids.reshape(-1, 1)).squeeze(1).float()
        logps = (selected - normalizer).reshape_as(completion_ids).masked_fill(~active, 0.0)
        entropy = None
        if compute_entropy:
            log_distribution = allowed_logits - normalizer[:, None]
            terms = log_distribution.masked_fill(~torch.isfinite(log_distribution), 0.0)
            entropy = -(log_distribution.exp() * terms).sum(-1).reshape_as(completion_ids)
            entropy = entropy.masked_fill(~active, 0.0)
        return logps, entropy
