"""Compare exact lap controls and CUDA latency with static cache compilation.

Uses saved dataset or evaluation JSONL prompts. This does not run the simulator.
Compile warmup is reported separately. Run on an idle GPU for useful timings.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from lap_policy import parse_action
from peft import PeftModel
from smoke_grpo import MODEL, REVISION
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.repeats, args.warmup) < 1:
        parser.error("samples, repeats and warmup must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.prompts.read_bytes()
    rows = [json.loads(line) for line in source.splitlines() if line.strip()]
    if not rows:
        raise ValueError("No prompts found")
    count = min(args.samples, len(rows))
    indices = [round(i * (len(rows) - 1) / max(count - 1, 1)) for i in range(count)]
    prompts = [rows[i]["prompt"] for i in indices]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
    lengths = [len(tokenizer(prompt)["input_ids"]) for prompt in prompts]
    if max(lengths) > 256:
        raise ValueError(f"Prompt exceeds fixed 256 token input: {max(lengths)}")
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
        ).cuda(),
        str(args.adapter),
    ).eval()
    model = model.merge_and_unload(safe_merge=True).eval()
    summary = {
        "model": MODEL,
        "revision": REVISION,
        "adapter_sha256": hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256(source).hexdigest(),
        "source_rows": indices,
        "prompt_lengths": lengths,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "dtype": "float32",
        "input_tokens_padded": 256,
        "cache_tokens": 288,
        "max_new_tokens": 32,
        "modes": {},
        "cuda_graph_mode_requested": "reduce-overhead",
        "cuda_graph_execution_verified": False,
    }
    baseline = None
    passed = True
    with (out / "generations.jsonl").open("w") as trace:
        for mode in ("eager", "eager_padded", "compiled_padded"):
            padded = mode != "eager"
            compiled = mode == "compiled_padded"
            encoded = [
                tokenizer(
                    prompt,
                    return_tensors="pt",
                    **({"padding": "max_length", "max_length": 256} if padded else {}),
                ).to("cuda")
                for prompt in prompts
            ]
            config = {
                "max_new_tokens": 32,
                "max_length": None,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "cache_implementation": "static" if compiled else "dynamic",
                "disable_compile": not compiled,
            }
            if compiled:
                config.update(
                    max_cache_len=288,
                    compile_config=CompileConfig(
                        # Gemma's sliding cache increments a Python position counter.
                        # Specializing that counter recompiles once per output token.
                        fullgraph=True,
                        dynamic=True,
                        mode="reduce-overhead",
                    ),
                )

            def generate(index, encoded=encoded, config=config):
                inputs = encoded[index]
                with torch.inference_mode():
                    result = model.generate(**inputs, **config)
                return result[0, inputs["input_ids"].shape[1] :].tolist()

            torch.cuda.synchronize()
            start = time.perf_counter()
            for i in range(args.warmup):
                generate(i % count)
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            timings, generations = [], []
            for repeat in range(args.repeats):
                for index in range(count):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    tokens = generate(index)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    try:
                        controls, error = parse_action(text), None
                    except ValueError as exc:
                        controls, error = None, str(exc)
                        passed = False
                    row = {
                        "mode": mode,
                        "repeat": repeat,
                        "source_row": indices[index],
                        "completion_ids": tokens,
                        "completion": text,
                        "controls": controls,
                        "error": error,
                        "seconds": elapsed,
                    }
                    generations.append(tokens)
                    timings.append(elapsed)
                    trace.write(json.dumps(row) + "\n")
                    trace.flush()
            if baseline is None:
                baseline = generations
            mismatches = sum(a != b for a, b in zip(baseline, generations, strict=True))
            passed &= mismatches == 0
            result = {
                "warmup_seconds": warmup_seconds,
                "calls": len(timings),
                "mean_seconds_per_control": statistics.mean(timings),
                "median_seconds_per_control": statistics.median(timings),
                "controls_per_second": len(timings) / sum(timings),
                "generated_tokens_per_second": sum(map(len, generations))
                / sum(timings),
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "token_mismatches_vs_eager": mismatches,
            }
            summary["modes"][mode] = result
            if compiled and args.profile:
                activities = torch.profiler.supported_activities()
                if torch.profiler.ProfilerActivity.CUDA not in activities:
                    summary["profile"] = {
                        "available": False,
                        "reason": "torch.profiler does not report CUDA tracing support",
                        "supported_activities": sorted(str(x) for x in activities),
                    }
                else:
                    torch.cuda.synchronize()
                    with (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as profiler,
                        torch.profiler.record_function("warmed_compiled_control"),
                    ):
                        profile_tokens = generate(0)
                        torch.cuda.synchronize()
                    profile_path = out / "compiled-control.trace.json"
                    profiler.export_chrome_trace(str(profile_path))
                    events = json.loads(profile_path.read_text())["traceEvents"]
                    graph_launches = [
                        event
                        for event in events
                        if event.get("cat") == "cuda_runtime"
                        and event.get("name", "").startswith("cudaGraphLaunch")
                    ]
                    kernel_events = sum(
                        event.get("cat") == "kernel" for event in events
                    )
                    matches = profile_tokens == baseline[0]
                    passed &= matches
                    summary["cuda_graph_execution_verified"] = bool(graph_launches)
                    summary["profile"] = {
                        "available": True,
                        "trace": profile_path.name,
                        "source_row": indices[0],
                        "completion_ids": profile_tokens,
                        "tokens_equal_eager": matches,
                        "cuda_graph_launch_events": len(graph_launches),
                        "cuda_kernel_events": kernel_events,
                        "reason": (
                            "CUDA runtime graph launch events recorded"
                            if graph_launches
                            else "No cudaGraphLaunch runtime events in profiler trace"
                        ),
                    }
            summary["all_controls_valid_and_tokens_equal"] = passed
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"mode": mode, **result}), flush=True)
    if not passed:
        raise SystemExit(
            "Generated controls failed validation or exact token equivalence"
        )


if __name__ == "__main__":
    main()
