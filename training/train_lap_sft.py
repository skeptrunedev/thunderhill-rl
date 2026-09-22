"""TRL supervised warm start for direct compact motorcycle controls."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from lap_policy import RoadTelemetry, parse_action
from model_runtime import (
    LEGACY_SPEC,
    GEMMA4_SPEC,
    GEMMA4_NATIVE_SPEC,
    PolicyRoadTelemetry,
    inference_precision,
    load_base,
    read_spec,
    write_spec,
)
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def prepare_dataset(path, tokenizer, spec, max_length):
    """Validate raw recorded telemetry, then tokenize once with explicit loss boundaries."""
    road = PolicyRoadTelemetry(spec, tokenizer)
    raw_road = RoadTelemetry()
    prepared = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        prompt = row["prompt"]
        if spec.prompt_style != "raw":
            try:
                features = json.loads(
                    prompt.split("\n", 1)[1].removesuffix("\nAction:\n")
                )
            except (ValueError, IndexError) as error:
                raise ValueError(
                    f"{path.name} row {index}: invalid recorded telemetry"
                ) from error
            if raw_road.prompt_features(features) != prompt:
                raise ValueError(
                    f"{path.name} row {index}: unrecognized telemetry prompt"
                )
            prompt = road.prompt_features(features)
        controls = parse_action(row["completion"])
        completion = (
            road.native_tools.completion_from_controls(controls)
            if spec.prompt_style == "gemma4_native_tools"
            else row["completion"] + tokenizer.eos_token
        )
        if spec.prompt_style == "gemma4_native_tools":
            if road.parse_completion(completion) != controls:
                raise ValueError(f"{path.name} row {index}: native conversion changed controls")
        kwargs = {} if spec.prompt_style == "raw" else {"add_special_tokens": False}
        prompt_ids = tokenizer(prompt, **kwargs)["input_ids"]
        tokens = tokenizer(prompt + completion, **kwargs)["input_ids"]
        if spec.prompt_style != "raw" and tokenizer(prompt)["input_ids"] != prompt_ids:
            raise ValueError(
                f"{path.name} row {index}: training and evaluation tokenization differ"
            )
        if tokens[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{path.name} row {index}: unstable prompt token boundary")
        if len(tokens) > max_length:
            raise ValueError(f"{path.name} row {index}: completion would be truncated")
        if spec.prompt_style != "raw" and tokens.count(tokenizer.bos_token_id) != 1:
            raise ValueError(f"{path.name} row {index}: expected exactly one chat BOS")
        if tokens[-1] != tokenizer.eos_token_id or len(tokens) <= len(prompt_ids):
            raise ValueError(f"{path.name} row {index}: missing completion terminator")
        prepared.append(
            {
                "input_ids": tokens,
                "completion_mask": [0] * len(prompt_ids)
                + [1] * (len(tokens) - len(prompt_ids)),
            }
        )
    if not prepared:
        raise ValueError(f"Empty dataset: {path}")
    return Dataset.from_list(prepared)


def verify_completion_labels(trainer, source):
    """Check the installed TRL preparation and collator preserve every loss boundary."""
    for expected, actual in zip(source, trainer.train_dataset, strict=True):
        labels = [
            token if keep else -100
            for token, keep in zip(
                expected["input_ids"], expected["completion_mask"], strict=True
            )
        ]
        if actual["input_ids"] != expected["input_ids"] or actual["labels"] != labels:
            raise ValueError("TRL altered supervised completion boundaries")
    sample = trainer.train_dataset[0]
    batch = trainer.data_collator([sample])
    if batch["labels"][0, : len(sample["labels"])].tolist() != sample["labels"]:
        raise ValueError("TRL collator altered supervised completion labels")


def training_spec(adapter, native_tools):
    source = read_spec(adapter) if adapter else LEGACY_SPEC
    if native_tools:
        if adapter is None or source not in (GEMMA4_SPEC, GEMMA4_NATIVE_SPEC):
            raise ValueError("Native tool migration requires a pinned Gemma4 source adapter")
        return GEMMA4_NATIVE_SPEC
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument(
        "--adapter", type=Path, help="Continue supervised training from this adapter"
    )
    p.add_argument("--native-tools", action="store_true",
                   help="Migrate a Gemma4 adapter to its native tool calling format")
    args = p.parse_args()
    spec = training_spec(args.adapter, args.native_tools)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    set_seed(71)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision)
    tokenizer.padding_side = "right"
    model = load_base(spec)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    max_length = (1024 if spec.prompt_style == "gemma4_native_tools"
                  else 256 if spec.prompt_style == "raw" else 512)
    train_data = prepare_dataset(
        args.dataset / "train.jsonl", tokenizer, spec, max_length
    )
    eval_data = prepare_dataset(
        args.dataset / "eval.jsonl", tokenizer, spec, max_length
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=SFTConfig(
            output_dir=str(out / "trainer"),
            max_steps=args.steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=1,
            max_length=max_length,
            completion_only_loss=True,
            loss_type="nll",
            bf16=spec.dtype == "bfloat16",
            fp16=False,
            gradient_checkpointing=False,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=None,
            logging_steps=10,
            report_to="none",
            seed=71,
        ),
        peft_config=None
        if args.adapter
        else LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0,
            task_type="CAUSAL_LM",
        ),
    )
    verify_completion_labels(trainer, train_data)
    started = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    write_spec(out / "adapter", spec)
    model = trainer.model
    model.eval()
    probe = torch.tensor([train_data[0]["input_ids"]], device="cuda")
    with torch.inference_mode(), inference_precision(model):
        expected = (
            model(input_ids=probe, logits_to_keep=1).logits.detach().float().cpu()
        )
    model.cpu()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(spec), out / "adapter").eval()
    with torch.inference_mode(), inference_precision(reloaded):
        actual = (
            reloaded(input_ids=probe, logits_to_keep=1).logits.detach().float().cpu()
        )
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)
    summary = {
        "method": "supervised warmstart, not RL",
        "model": spec.model,
        "revision": spec.revision,
        "prompt_style": spec.prompt_style,
        "source_prompt_style": read_spec(args.adapter).prompt_style if args.adapter else None,
        "maximum_train_tokens": max(map(len, train_data["input_ids"])),
        "maximum_eval_tokens": max(map(len, eval_data["input_ids"])),
        "supervision_controls_preserved": True,
        "reloaded_logits_match": True,
        "completion_masks_verified": True,
        "max_length": max_length,
        "learning_rate": args.learning_rate,
        "steps": trainer.state.global_step,
        "adapter_sha256": hashlib.sha256(
            (out / "adapter" / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "eval_data_sha256": hashlib.sha256(
            (args.dataset / "eval.jsonl").read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest()
        if (args.dataset / "manifest.json").exists()
        else None,
        "initial_adapter_sha256": hashlib.sha256(
            (args.adapter / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        if args.adapter
        else None,
        "seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "metrics": result.metrics,
        "train_data_sha256": hashlib.sha256(
            (args.dataset / "train.jsonl").read_bytes()
        ).hexdigest(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
