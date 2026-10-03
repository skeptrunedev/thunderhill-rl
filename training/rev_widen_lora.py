"""Widen a Rev checkpoint's LoRA to a higher rank without changing what it computes.

Kev builds its LoRA with lora_alpha = 2 x rank, so the scale alpha / rank is 2 at every rank
and each adapted weight moves by 2 * B @ A. Widening A (rank x in) with fresh rows and B
(out x rank) with zero columns leaves B @ A, and so every output, exactly as it was, as
peft's own initialisation does (random A, zero B); training then grows the new directions.
The pointer head, temperature and results are copied, and head.pt records the new rank so
`kev.train --lora RANK --init_from` accepts it.

  uv run --project ~/git_projects/references/jev-models/kev python training/rev_widen_lora.py \\
      --run rev-0.8b-s9 --rank 64 --name rev-0.8b-s9w64
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from kev.checkpoint import read_meta, write_meta

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "runs/rev"


def widen(tensors: dict, rank: int, generator: torch.Generator) -> dict:
    out = {}
    for key, value in tensors.items():
        if ".lora_A." in key:
            old, fan_in = value.shape
            extra = torch.empty(rank - old, fan_in, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(extra, a=math.sqrt(5), generator=generator)  # peft's lora_A init
            out[key] = torch.cat([value, extra.to(value.dtype)], dim=0)
        elif ".lora_B." in key:
            fan_out, old = value.shape
            out[key] = torch.cat([value, torch.zeros(fan_out, rank - old, dtype=value.dtype)], dim=1)
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="a completed Rev run in runs/rev")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--name", required=True, help="the widened run's name")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source, target = REV / args.run, REV / args.name
    config = json.loads((source / "checkpoint/adapter_config.json").read_text())
    if args.rank <= config["r"] or config["lora_alpha"] != 2 * config["r"] or config.get("use_rslora"):
        raise SystemExit(f"need a larger rank than {config['r']} and Kev's alpha = 2 x rank (no rsLoRA)")
    if target.exists():
        raise SystemExit(f"{target} exists")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("calibration", "development", "data"))
    checkpoint = target / "checkpoint"

    tensors = load_file(checkpoint / "adapter_model.safetensors")
    wide = widen(tensors, args.rank, torch.Generator().manual_seed(args.seed))
    # The function is unchanged: every adapted weight's update B @ A is identical.
    for key in [k for k in tensors if ".lora_A." in k]:
        b = key.replace(".lora_A.", ".lora_B.")
        before = tensors[b].float() @ tensors[key].float()
        after = wide[b].float() @ wide[key].float()
        if not torch.allclose(before, after, atol=1e-6):
            raise RuntimeError(f"widening changed {key}")
    save_file(wide, checkpoint / "adapter_model.safetensors", metadata={"format": "pt"})
    config.update(r=args.rank, lora_alpha=2 * args.rank)
    (checkpoint / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    meta = read_meta(str(checkpoint))
    meta.lora = args.rank
    meta.extra["widened"] = {"from": args.run, "rank": args.rank, "seed": args.seed}
    write_meta(str(checkpoint), meta)

    result = json.loads((target / "result.json").read_text())
    result.update(name=args.name, widened_from=args.run, lora_rank=args.rank)
    (target / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    trainable = sum(v.numel() for k, v in wide.items() if ".lora_" in k)
    print(f"{args.name}: {args.run}'s adapter widened from rank {tensors and next(v.shape[0] for k, v in tensors.items() if '.lora_A.' in k)} "
          f"to {args.rank} ({trainable / 1e6:.1f}M adapter weights), outputs unchanged")


if __name__ == "__main__":
    main()
