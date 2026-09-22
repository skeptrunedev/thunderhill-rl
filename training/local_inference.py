"""Resident CPU embedding lookup for unquantized Gemma4 on a small CUDA GPU.

Only looked up embedding vectors cross PCIe. Accelerate's ordinary CPU weight
offload would copy the entire 5.25 GiB table onto CUDA for every forward pass.
FP16 computation differs from the H100 BF16 training runtime and is reported.
"""

import hashlib
import json
import time

import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from peft import PeftModel
from transformers import LogitsProcessor

from lap_policy import parse_action
from model_runtime import GEMMA4_NATIVE_SPEC, GEMMA4_SPEC, load_base


class FiniteLogits(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Local FP16 generation produced nonfinite logits")
        return scores


def load_local_policy(spec, adapter):
    if spec not in (GEMMA4_SPEC, GEMMA4_NATIVE_SPEC):
        raise ValueError("The local FP16 profile requires the pinned Gemma4 model")
    if not torch.cuda.is_available():
        raise ValueError("The local FP16 profile requires CUDA")
    base = load_base(spec, device="cpu", dtype="float16")
    # Explicit placement prevents PEFT's default auto dispatch from moving
    # the CPU embedding table back onto the GPU while loading the adapter.
    model = PeftModel.from_pretrained(
        base, str(adapter), torch_device="cpu", device_map={"": "cpu"},
        autocast_adapter_dtype=True,
    ).eval()
    place_resident_embedding(base, "cuda:0")
    return model


def place_resident_embedding(base, device):
    for name, module in base.model.named_children():
        if name != "embed_tokens_per_layer":
            module.to(device)
    # Preserve the tied vocabulary input/output weights.
    base.lm_head.to(device)
    if base.lm_head.weight.data_ptr() != base.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("Vocabulary weights lost their shared storage")
    embedding = base.model.embed_tokens_per_layer
    add_hook_to_module(
        embedding,
        AlignDevicesHook(execution_device="cpu", offload=False, io_same_device=True),
    )
    base.hf_device_map = {"model." + name: ("cpu" if name == "embed_tokens_per_layer" else str(device))
                          for name, _ in base.model.named_children()}
    base.hf_device_map["lm_head"] = str(device)


def compare_reference(model, tokenizer, path, adapter_hash, count=12):
    """Measure precision differences on recorded prompts, without applying controls."""
    if count < 1:
        raise ValueError("Reference sample count must be positive")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(row["adapter_sha256"] != adapter_hash for row in rows):
        raise ValueError("Reference decisions must belong to this exact adapter")
    samples = min(count, len(rows))
    indices = sorted({round(i * (len(rows) - 1) / (samples - 1))
                      for i in range(samples)}) if samples > 1 else [0]
    results = []
    started = time.monotonic()
    native = tokenizer.eos_token_id == 50
    if native:
        from native_tools import NativeBikeTools
        from native_constraints import NativeToolConstraint
        tools = NativeBikeTools(tokenizer)
        constraints = NativeToolConstraint(tokenizer, model.config.vocab_size)
    for index in indices:
        row = rows[index]
        inputs = tokenizer(row["prompt"], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite logits in local inference")
            output = model.generate(
                **inputs, max_new_tokens=tools.max_completion_length if native else 32, max_length=None, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                logits_processor=[FiniteLogits(), constraints.logits_processor()] if native else [FiniteLogits()],
            )
        text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=not native)
        parse = tools.parse_completion if native else parse_action
        controls = parse(text)
        expected = parse(row["completion"])
        results.append({"source_action_index": row["action_index"],
                        "reference": row["completion"], "local": text,
                        "controls_match": controls == expected})
    return {"reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "adapter_sha256": adapter_hash, "samples": len(results),
            "exact_control_matches": sum(row["controls_match"] for row in results),
            "finite_logits": True, "all_controls_valid": True,
            "seconds": time.monotonic() - started, "results": results,
            "limitation": "FP16 local inference versus recorded H100 BF16 decisions; this is not numerical equivalence."}
