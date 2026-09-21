"""TRL supervised warm start for direct compact motorcycle controls."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from smoke_grpo import MODEL, REVISION
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=4)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    set_seed(71)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, attn_implementation="sdpa"
    )

    def load(name):
        rows = [
            json.loads(line) for line in (args.dataset / name).read_text().splitlines()
        ]
        return Dataset.from_list(
            [
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"] + tokenizer.eos_token,
                }
                for row in rows
            ]
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=load("train.jsonl"),
        eval_dataset=load("eval.jsonl"),
        args=SFTConfig(
            output_dir=str(out / "trainer"),
            max_steps=args.steps,
            learning_rate=2e-4,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=1,
            max_length=256,
            completion_only_loss=True,
            loss_type="nll",
            bf16=False,
            fp16=False,
            gradient_checkpointing=False,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            logging_steps=10,
            report_to="none",
            seed=71,
        ),
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0,
            task_type="CAUSAL_LM",
        ),
    )
    started = time.monotonic()
    result = trainer.train()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    summary = {
        "method": "supervised warmstart, not RL",
        "model": MODEL,
        "revision": REVISION,
        "steps": trainer.state.global_step,
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
