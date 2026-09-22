"""CPU tests for exact prompt memoization and frozen policy scope boundaries."""

import unittest

from greedy_cache import GreedyGenerationCache


class GreedyCacheTests(unittest.TestCase):
    def test_exact_prompt_generated_once_and_distinct_prompt_not_aliased(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return prompt, [len(calls)]

        cache = GreedyGenerationCache(generate, scope="step-0")
        self.assertEqual(cache("speed=1"), ("speed=1", [1]))
        self.assertEqual(cache("speed=1"), ("speed=1", [1]))
        self.assertEqual(cache("speed=1 "), ("speed=1 ", [2]))
        self.assertEqual(calls, ["speed=1", "speed=1 "])
        self.assertEqual((cache.hits, cache.misses), (1, 2))

    def test_new_step_and_evaluation_never_reuse_old_policy_outputs(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return f"policy-{len(calls)}", [len(calls)]

        for index, scope in enumerate(("baseline", "step-0", "step-1", "after"), 1):
            cache = GreedyGenerationCache(generate, scope=scope)
            self.assertEqual(cache("same"), (f"policy-{index}", [index]))
            self.assertEqual(cache("same"), (f"policy-{index}", [index]))
        self.assertEqual(len(calls), 4)

    def test_generator_and_caller_mutation_cannot_corrupt_cached_ids(self):
        original = [1, 2]
        cache = GreedyGenerationCache(lambda prompt: ("text", original), scope="step")
        text, ids = cache("prompt")
        original.append(3)
        ids[0] = 999
        self.assertEqual(cache("prompt"), (text, [1, 2]))
        another = cache("prompt")[1]
        another.clear()
        self.assertEqual(cache("prompt"), (text, [1, 2]))

    def test_generation_error_propagates_without_populating_cache(self):
        def fail(prompt):
            raise RuntimeError("generation failed")

        cache = GreedyGenerationCache(fail, scope="step")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            cache("prompt")
        self.assertEqual((cache.hits, cache.misses), (0, 0))


if __name__ == "__main__":
    unittest.main()
