"""Memoize deterministic text generation within one frozen policy scope.

Create a fresh instance for every optimizer step or evaluation. This caches only
model output for identical prompts, never simulator state or transitions.
"""


class GreedyGenerationCache:
    def __init__(self, generate, *, scope):
        self.generate = generate
        self.scope = scope
        self.hits = 0
        self.misses = 0
        self._outputs = {}

    def __call__(self, prompt):
        if prompt in self._outputs:
            self.hits += 1
        else:
            text, ids = self.generate(prompt)
            # Own an immutable copy, so neither the generator nor a caller can
            # mutate a result subsequently reused by another candidate rollout.
            self._outputs[prompt] = (text, tuple(ids))
            self.misses += 1
        text, ids = self._outputs[prompt]
        return text, list(ids)
