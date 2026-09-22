"""Batched sampled bike decisions with their exact behavior token likelihoods."""

from __future__ import annotations

import gc
import json
import math
import time

import torch
from transformers import CompileConfig, LogitsProcessor, LogitsProcessorList

from lap_policy import parse_action
from model_runtime import inference_precision


class GreedyRows(LogitsProcessor):
    """Validate raw logits, then constrain only explicitly excluded eval rows."""

    def __init__(self, indices=()):
        self.indices = tuple(indices)

    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite batched model logits")
        if self.indices:
            rows = torch.tensor(self.indices, device=scores.device)
            chosen = scores[rows].argmax(dim=-1)
            scores = scores.clone()
            scores[rows] = -torch.inf
            scores[rows, chosen] = 0
        return scores


class BatchedPolicy:
    def __init__(self, model, tokenizer, *, compile_inference=False, compiled_prompt_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.compile_inference = compile_inference
        self.compile_qualified = False
        self.compiled_prompt_length = compiled_prompt_length
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Explicit padding and EOS token IDs are required")
        if compile_inference and model.device.type != "cuda":
            raise ValueError("Compiled generation requires CUDA")
        if compiled_prompt_length <= 0:
            raise ValueError("Compiled prompt length must be positive")
        tokenizer.padding_side = "left"

    def _qualify_compilation(self, generated_steps):
        if not self.compile_inference or generated_steps <= 1:
            return
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        # Transformers 5.17 installs this callable only through get_compiled_call.
        # A static cache request alone is insufficient: HF can warn and skip it.
        if not callable(getattr(base, "_compiled_call", None)):
            raise RuntimeError("Requested compiled inference was skipped by Transformers")
        self.compile_qualified = True

    def generate(self, prompts: list[str], greedy_indices=()) -> list[dict]:
        if not prompts or any(not isinstance(p, str) or not p for p in prompts):
            raise ValueError("A nonempty list of nonempty prompts is required")
        greedy_indices = tuple(greedy_indices)
        if len(set(greedy_indices)) != len(greedy_indices) or any(
            type(i) is not int or not 0 <= i < len(prompts) for i in greedy_indices
        ):
            raise ValueError("Greedy row indices must be unique and within the batch")
        tokenizer = self.tokenizer
        inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if self.compile_inference:
            if inputs["input_ids"].shape[1] > self.compiled_prompt_length:
                raise ValueError("Prompt exceeds compiled width; prompts are never truncated")
            inputs = tokenizer(prompts, padding="max_length", max_length=self.compiled_prompt_length,
                               truncation=False, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        width = inputs["input_ids"].shape[1]
        options = {}
        if self.compile_inference:
            options = dict(cache_implementation="static", max_cache_len=width + 32,
                           compile_config=CompileConfig(fullgraph=True, dynamic=True, mode="reduce-overhead"))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode(), inference_precision(self.model):
                output = self.model.generate(
                    **inputs, max_new_tokens=32, max_length=None, do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    use_cache=True, output_scores=True, return_dict_in_generate=True,
                    logits_processor=LogitsProcessorList([GreedyRows(greedy_indices)]), **options,
                )
        finally:
            self.model.train(was_training)
        self._qualify_compilation(len(output.scores))
        tokens = output.sequences[:, width:]
        if tokens.shape[1] != len(output.scores):
            raise ValueError("Generated token and behavior score lengths disagree")
        # Reduce each score matrix immediately, never stack B x T x vocabulary.
        logps = []
        for step, scores in enumerate(output.scores):
            selected = scores.float().log_softmax(-1).gather(1, tokens[:, step:step + 1]).squeeze(1)
            logps.append(selected)
        logps = torch.stack(logps).cpu().tolist()
        result = []
        for row in range(len(prompts)):
            ids = tokens[row].tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
            likelihoods = [values[row] for values in logps[:len(ids)]]
            if not all(math.isfinite(value) for value in likelihoods):
                raise FloatingPointError("Nonfinite selected behavior log probability")
            result.append({
                "prompt_ids": inputs["input_ids"][row][inputs["attention_mask"][row].bool()].tolist(),
                "completion_ids": ids,
                "completion": tokenizer.decode(ids, skip_special_tokens=True),
                "old_per_token_logps": likelihoods,
            })
        return result

    def profile(self, prompts, candidates=(4, 8, 16, 32, 64, 128, 256), headroom_fraction=.15):
        """Qualify batch sizes using real prompts; does not change weights or RNG."""
        if self.model.device.type != "cuda":
            raise ValueError("Memory profiling requires CUDA")
        if not prompts or not 0 < headroom_fraction < 1:
            raise ValueError("Prompts and a headroom fraction between zero and one are required")
        if not candidates or any(type(n) is not int or n <= 0 for n in candidates):
            raise ValueError("Batch candidates must be positive integers")
        device = self.model.device
        records = []
        with torch.random.fork_rng(devices=[device.index or 0]):
            for size in candidates:
                batch = [prompts[i % len(prompts)] for i in range(size)]
                record = {"batch_size": size, "repeats": 2}
                print(json.dumps({"event": "batch_profile_start", "batch_size": size,
                                  "compile_inference": self.compile_inference}), flush=True)
                warmup_started = time.perf_counter()
                try:
                    self.generate(batch)  # Warm kernels and the actual generation shape.
                    torch.cuda.synchronize(device)
                    record["warmup_seconds"] = time.perf_counter() - warmup_started
                    print(json.dumps({"event": "batch_profile_warmed", "batch_size": size,
                                      "warmup_seconds": record["warmup_seconds"]}), flush=True)
                    torch.cuda.reset_peak_memory_stats(device)
                    started = time.perf_counter()
                    valid, tokens = 0, 0
                    for _ in range(2):
                        outputs = self.generate(batch)
                        for row in outputs:
                            tokens += len(row["completion_ids"])
                            try:
                                parse_action(row["completion"])
                                valid += 1
                            except ValueError:
                                pass
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    free, total = torch.cuda.mem_get_info(device)
                    peak = torch.cuda.max_memory_reserved(device)
                    # Account for device allocations outside PyTorch as well.
                    external = max(0, total - free - torch.cuda.memory_reserved(device))
                    headroom = max(0, total - external - peak) / total
                    record.update(
                        elapsed_seconds=elapsed, valid_controls=valid,
                        valid_calls_per_second=valid / elapsed, calls_per_second=2 * size / elapsed,
                        tokens_per_second=tokens / elapsed,
                        peak_reserved_bytes=peak,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        free_bytes=free, total_bytes=total, headroom_fraction=headroom,
                        compile_qualified=self.compile_qualified,
                        eligible=(headroom >= headroom_fraction and valid > 0
                                  and (not self.compile_inference or self.compile_qualified)),
                    )
                except torch.cuda.OutOfMemoryError:
                    record.update(error="cuda_out_of_memory", eligible=False)
                    records.append(record)
                    print(json.dumps({"event": "batch_profile_result", **record}), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                records.append(record)
                print(json.dumps({"event": "batch_profile_result", **record}), flush=True)
        eligible = [r for r in records if r.get("eligible")]
        best = max(eligible, key=lambda r: r["valid_calls_per_second"], default=None)
        return {"selected_batch_size": best["batch_size"] if best else None,
                "headroom_required": headroom_fraction, "compile_inference": self.compile_inference,
                "compile_qualified": self.compile_qualified,
                "results": records}
