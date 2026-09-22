"""Measure valid control exploration before spending time on simulator rollouts."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from lap_policy import parse_action
from peft import PeftModel
from smoke_grpo import MODEL, REVISION
from train_lap_grpo import checkpoint_hash
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--action-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[1.4, 1.8, 2.2]
    )
    args = parser.parse_args()
    if (
        args.samples < 1
        or args.action_index < 0
        or any(not math.isfinite(t) or t <= 0 for t in args.temperatures)
    ):
        parser.error(
            "Require positive sample count and temperatures, nonnegative action index"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    row = json.loads(args.decisions.read_text().splitlines()[args.action_index])
    fingerprint = checkpoint_hash(args.adapter)
    if row["adapter_sha256"] != fingerprint or row["action_index"] != args.action_index:
        raise ValueError("Source prompt and adapter identity do not match")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
        ).cuda(),
        str(args.adapter),
    ).eval()
    inputs = tokenizer(
        [row["prompt"]] * args.samples, return_tensors="pt", padding=True
    ).to("cuda")
    results = []
    for temperature in args.temperatures:
        set_seed(71)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                max_length=None,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        controls, invalid = [], []
        for text in texts:
            try:
                controls.append(parse_action(text))
            except ValueError:
                invalid.append(text)
        result = {
            "temperature": temperature,
            "samples": len(texts),
            "valid": len(controls),
            "unique_valid": len({json.dumps(c, sort_keys=True) for c in controls}),
            "throttle_values": sorted({c["throttle"] for c in controls}),
            "completions": dict(Counter(texts)),
            "invalid_completions": invalid,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(
            {
                "model": MODEL,
                "adapter_sha256": fingerprint,
                "source_decisions": str(args.decisions),
                "action_index": args.action_index,
                "prompt": row["prompt"],
                "seed": 71,
                "results": results,
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
