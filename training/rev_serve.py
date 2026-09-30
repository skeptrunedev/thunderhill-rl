"""Serve Rev on the local GPU: kev.serve with batching sized for Rev's short states.

kev.serve answers requests one at a time unless its CUDA graphs are on: the graphed
passes are what batch every waiting request into one state pass and one row pass
(kev.cuda_graphs). Their buffers are sized for 64k-token documents (a 4096-position
bank entry per state, 32 rows of DeltaNet states), several GB, so under the shared-GPU
cap they did not fit next to fp32 weights and rev_dagger served eagerly: 5-9 requests/s
on the 2080 Ti however many envs asked. Rev's requests are small and uniform: a state of
about 340 tokens (at most kev.model.MAX_STATE = 384) and three question rows of at most
~50 tokens, one request per env. So the limits here hold one batch of --max-batch such
requests (a larger batch runs as several graphed passes), and the whole server stays
inside --gpu-memory-gb.

The weights run in fp16: the 2080 Ti (Turing) has fp16 tensor cores but no bf16, and fp32
halves the batch the cap admits. Against fp32 on 64 development records of rev-0.8b-r8,
fp16 with graphs moved probabilities by at most 0.003 and flipped no argmax (192 answers).
Every state is new (a ride never repeats one), so the state-prefix cache is off.

Measured on rev-0.8b-r8, batches of 16 development requests: 20 requests/s with the GPU
to itself, about 10 while a SAC learner holds it at 100% (5-7 eager in fp32).

Run it in the pinned Kev checkout's environment (rev_dagger.Server does):

  cd ~/git_projects/references/jev-models/kev && uv run --extra serve python \\
      THUNDERHILL/training/rev_serve.py --run THUNDERHILL/runs/rev/rev-0.8b-r8/checkpoint --port 8019
"""

from __future__ import annotations

import argparse
import math
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="a Rev checkpoint directory")
    parser.add_argument("--port", type=int, default=8019)
    parser.add_argument("--gpu-memory-gb", type=float, default=4.5,
                        help="cap on this process's CUDA allocations (its CUDA context adds ~0.5 GB)")
    parser.add_argument("--max-batch", type=int, default=8,
                        help="requests one graphed pass holds; a larger batch runs as several passes")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp16")
    parser.add_argument("--fused", type=int, choices=(0, 1), default=0,
                        help="use Kev's fused Qwen kernels when its pinned FLA package is installed")
    parser.add_argument("--temperature", type=float,
                        help="override checkpoint temperature; 1 preserves the learned soft action targets")
    args = parser.parse_args()
    if args.max_batch < 1:
        parser.error("--max-batch must be at least 1")
    if args.fused and args.dtype != "bf16":
        parser.error("Kev's fused Qwen kernels require --dtype bf16")
    if args.temperature is not None:
        if not math.isfinite(args.temperature) or args.temperature <= 0:
            parser.error("--temperature must be finite and positive")
        os.environ["KEV_TEMPERATURE"] = str(args.temperature)
    # Before torch loads: a batch's shapes vary, and fixed-size segments fragment under the cap.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.update(KEV_DTYPE=args.dtype, KEV_CUDA_GRAPHS="1", KEV_FUSED=str(args.fused), KEV_PREFIX_CACHE="0")

    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_memory_gb * 2**30 / total))

    import kev.cuda_graphs as graphs
    from kev.model import MAX_STATE

    questions, row = 3, 128
    graphs.GRAPH_STATES = args.max_batch                 # states per state pass: the bank's entries
    graphs.BANK_WIDTH = graphs.GRAPH_STATE = 2 * MAX_STATE  # positions per state, bucketed up from MAX_STATE
    graphs.GRAPH_ROWS = questions * args.max_batch        # question rows per row pass
    graphs.GRAPH_ROW = row                                # longest question-row bucket
    # Each question row attends to its telemetry prefix too. Even the smallest
    # batch must fit one complete row; larger batches split into buffered passes.
    graphs.GRAPH_TOKENS = max(graphs.GRAPH_ROWS * row, graphs.pow2(MAX_STATE) + row)

    from uvicorn.config import LOGGING_CONFIG

    LOGGING_CONFIG["loggers"]["uvicorn.access"]["level"] = "WARNING"  # a line per control step otherwise
    sys.argv = ["kev.serve", "--run", args.run, "--port", str(args.port)]
    runpy.run_module("kev.serve", run_name="__main__")


if __name__ == "__main__":
    main()
