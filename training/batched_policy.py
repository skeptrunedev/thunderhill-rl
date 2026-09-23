"""Batched sampled bike decisions with their exact behavior token likelihoods."""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import time

import torch
from transformers import CompileConfig, LogitsProcessor, LogitsProcessorList

from lap_policy import parse_action
from model_runtime import inference_precision


class GreedyRows(LogitsProcessor):
    """Validate raw logits, then constrain only explicitly excluded eval rows."""

    def __init__(self, indices=(), *, validate_raw=True):
        self.indices = tuple(indices)
        self.validate_raw = validate_raw

    def __call__(self, input_ids, scores):
        if self.validate_raw and not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite batched model logits")
        if self.indices:
            rows = torch.tensor(self.indices, device=scores.device)
            chosen = scores[rows].argmax(dim=-1)
            scores = scores.clone()
            scores[rows] = -torch.inf
            scores[rows, chosen] = 0
        return scores


class FiniteModelScores(LogitsProcessor):
    """Check model outputs before a grammar intentionally inserts negative inf."""

    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Nonfinite batched model logits")
        return scores


def cuda_graph_evidence(device):
    """Read PyTorch 2.14 tree state without creating managers or capturing work.

    Counts are process cumulative, not proof every model operation was captured.
    The manager's own graph is an empty allocator graph and is excluded.
    """
    try:
        from torch._dynamo.utils import counters
        from torch._inductor.cudagraph_trees import get_manager
        manager = get_manager(device.index or 0, create_if_none_exists=False)
        pending = [] if manager is None else [node for nodes in manager.roots.values() for node in nodes]
        seen, graphs = set(), 0
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            graphs += int(getattr(node, "graph", None) is not None)
            pending.extend(child for children in node.children.values() for child in children)
        path = getattr(manager, "path_state", None)
        return {
            "available": True, "scope": "process_cumulative",
            "captured_nodes": graphs, "cuda_graph_captured": graphs > 0,
            "skips": int(counters.get("inductor", {}).get("cudagraph_skips", 0)),
            "path_state": getattr(path, "name", None),
        }
    except (ImportError, AttributeError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}


def cuda_memory_evidence(device):
    """Separate allocator memory from driver allocations and other processes.

    External bytes include both other processes and nonallocator allocations in
    this process; they must not be attributed to a leak without process evidence.
    This is diagnostic only and never resets compiled graphs or allocator state.
    """
    if device.type != "cuda":
        return {"available": False, "reason": "non_cuda_device"}
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    report = dict(
        available=True, pid=os.getpid(), free_bytes=free, total_bytes=total,
        allocated_bytes=torch.cuda.memory_allocated(device), reserved_bytes=reserved,
        nonallocator_device_bytes=max(0, total - free - reserved),
        cuda_graph_evidence=cuda_graph_evidence(device),
    )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        report["process_memory_error"] = "nvidia-smi is unavailable"
    else:
        try:
            result = subprocess.run(
                [executable, "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                timeout=10, check=True,
            )
            report["compute_process_memory_columns"] = ["pid", "used_gpu_memory_mib"]
            report["compute_process_memory_csv"] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            report["process_memory_error"] = f"{type(error).__name__}: {error}"
    return report


class BatchedPolicy:
    def __init__(self, model, tokenizer, *, compile_inference=False, compiled_prompt_length=None,
                 native_tools=None, constraints=None, temperature=1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        self.temperature = float(temperature)
        self.model = model
        self.tokenizer = tokenizer
        self.compile_inference = compile_inference
        self.compile_qualified = False
        self.native_tools = native_tools
        if native_tools is not None and constraints is None:
            from native_constraints import NativeToolConstraint
            constraints = NativeToolConstraint(tokenizer, model.config.vocab_size, native_tools=native_tools)
        if constraints is not None and native_tools is None:
            raise ValueError("Native constraints require the native tool protocol")
        self.constraints = constraints
        self.max_completion_length = native_tools.max_completion_length if native_tools else 32
        self.compiled_prompt_length = compiled_prompt_length if compiled_prompt_length is not None else (1024 if native_tools else 256)
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("Explicit padding and EOS token IDs are required")
        if compile_inference and model.device.type != "cuda":
            raise ValueError("Compiled generation requires CUDA")
        if self.compiled_prompt_length <= 0:
            raise ValueError("Compiled prompt length must be positive")
        tokenizer.padding_side = "left"

    @property
    def stop_token_id(self):
        return self.native_tools.stop_token_id if self.native_tools else self.tokenizer.eos_token_id

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
        tokenize_options = {"add_special_tokens": False} if self.native_tools else {}
        inputs = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt", **tokenize_options)
        if self.compile_inference:
            if inputs["input_ids"].shape[1] > self.compiled_prompt_length:
                raise ValueError("Prompt exceeds compiled width; prompts are never truncated")
            inputs = tokenizer(prompts, padding="max_length", max_length=self.compiled_prompt_length,
                               truncation=False, return_tensors="pt", **tokenize_options)
        inputs = inputs.to(self.model.device)
        width = inputs["input_ids"].shape[1]
        options = {}
        if self.compile_inference:
            options = dict(cache_implementation="static", max_cache_len=width + self.max_completion_length,
                           compile_config=CompileConfig(fullgraph=True, dynamic=True, mode="reduce-overhead"))
        processors = [GreedyRows(greedy_indices)]
        if self.constraints is not None:
            # Finite checks inspect raw logits. The grammar then masks illegal
            # choices, and greedy evaluation selects among legal choices only.
            processors = [FiniteModelScores(), self.constraints.logits_processor(),
                          GreedyRows(greedy_indices, validate_raw=False)]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode(), inference_precision(self.model):
                output = self.model.generate(
                    **inputs, max_new_tokens=self.max_completion_length, max_length=None, do_sample=True,
                    temperature=self.temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=self.stop_token_id,
                    use_cache=True, output_scores=True, return_dict_in_generate=True,
                    logits_processor=LogitsProcessorList(processors), **options,
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
            if self.stop_token_id in ids:
                ids = ids[:ids.index(self.stop_token_id) + 1]
            likelihoods = [values[row] for values in logps[:len(ids)]]
            if not all(math.isfinite(value) for value in likelihoods):
                raise FloatingPointError("Nonfinite selected behavior log probability")
            completion = tokenizer.decode(ids, skip_special_tokens=self.native_tools is None)
            if self.native_tools is not None:
                self.native_tools.parse_completion(completion)
            result.append({
                "prompt_ids": inputs["input_ids"][row][inputs["attention_mask"][row].bool()].tolist(),
                "completion_ids": ids,
                "completion": completion,
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
                graphs_before = cuda_graph_evidence(device)
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
                                (self.native_tools.parse_completion if self.native_tools else parse_action)(row["completion"])
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
                    graphs_after = cuda_graph_evidence(device)
                    record["cuda_graph_evidence"] = graphs_after
                    if graphs_before.get("available") and graphs_after.get("available"):
                        record["cuda_graph_nodes_added"] = graphs_after["captured_nodes"] - graphs_before["captured_nodes"]
                        record["cuda_graph_skips_added"] = graphs_after["skips"] - graphs_before["skips"]
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
                "compiled_prompt_length": self.compiled_prompt_length,
                "compile_qualified": self.compile_qualified,
                "results": records}
