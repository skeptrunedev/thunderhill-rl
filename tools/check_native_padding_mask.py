"""Numerically audit the reviewed native prefix mask with real PyTorch SDPA.

This is a CPU tensor correctness check, not gameplay or a training experiment.
It does not claim that the separate CUDA scatter failure has been resolved.
"""

import argparse
import json
from pathlib import Path
import runpy

import torch


def check(root: Path) -> dict:
    from training.native_recipe_source import verify_recipe

    provenance = verify_recipe(root)
    functions = runpy.run_path(
        str(root / "alpamayo1_x_rl/models/expert_model/padding_mask.py")
    )
    prompt = functions["prompt_mask"]
    extend = functions["extend_prompt_mask"]
    mask_prefix = functions["mask_expert_prefix"]
    ids = torch.tensor([[10, 11, 12, 13, 14], [0, 0, 10, 11, 12]])
    padding = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    mask = prompt(ids, padding)
    assert torch.equal(mask, padding.bool())
    assert torch.equal(prompt(ids, None), torch.ones_like(ids, dtype=torch.bool))
    extended = extend(mask, 7)
    assert torch.equal(extended[:, :5], mask) and extended[:, 5:].all()
    generator = torch.Generator().manual_seed(901)
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        for chunks in (1, 3):
            queries = chunks * 2
            q = torch.randn(2, 1, queries, 8, generator=generator).to(dtype)
            k = torch.randn(2, 1, 5 + queries, 8, generator=generator).to(dtype)
            v = torch.randn(2, 1, 5 + queries, 8, generator=generator).to(dtype)
            # Deliberately large padding content must have no effect.
            k[1, :, :2] = 20
            v[1, :, :2] = -20
            suffix = torch.arange(queries) // 2
            native = torch.zeros(2, 1, queries, 5 + queries, dtype=dtype)
            native[..., 5:].masked_fill_(
                suffix[:, None] != suffix[None, :], float("-inf")
            )
            suffix_before = native[..., 5:].clone()
            padded = mask_prefix(native.clone(), mask)
            assert torch.equal(padded[..., 5:], suffix_before)
            assert torch.equal(padded[0], native[0])
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=padded
            )
            for row in range(2):
                keep = torch.cat((mask[row], torch.ones(queries, dtype=torch.bool)))
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q[row : row + 1],
                    k[row : row + 1, :, keep],
                    v[row : row + 1, :, keep],
                    attn_mask=native[row : row + 1, :, :, keep],
                )
                torch.testing.assert_close(
                    actual[row : row + 1],
                    reference,
                    atol=0.02 if dtype == torch.bfloat16 else 1e-6,
                    rtol=0.01 if dtype == torch.bfloat16 else 1e-6,
                )
            repeated = mask_prefix(
                native.repeat_interleave(2, 0), mask.repeat_interleave(2, 0)
            )
            assert torch.equal(repeated, padded.repeat_interleave(2, 0))
            cases.append(
                dict(
                    dtype=str(dtype),
                    replay_chunks=chunks,
                    padded_batch_matches_unpadded_singletons=True,
                    suffix_mask_preserved=True,
                    sample_repeat_preserved=True,
                )
            )
    return dict(
        provenance=provenance,
        cases=cases,
        cpu_sdpa_verified=True,
        full_model_verified=False,
        cuda_scatter_root_cause_verified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
