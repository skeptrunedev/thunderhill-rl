"""Serve one Rev checkpoint from Modal: the /v1/systemone endpoint rev_drive.py rides with.

The checkpoint is a run on modal_rev.py's volume (REV_SERVE_RUN, fixed at deploy time);
kev.serve runs it on an H100 in bf16 with CUDA graphs, behind a bearer key from the
Modal secret rev-serve-key (KEV_API_KEY). Serving remotely keeps the local GPU free for
SAC training, and an H100 answers a DAgger collection several times faster than the
local 2080 Ti in fp32.

One container, never more: at 20 concurrent requests Modal's autoscaler started a second
H100 (well under max_inputs), and requests routed to it waited out its cold start. One
container answered 35 requests/s at 20 concurrent clients and 61 at 48 (rev-0.8b-r8, from
the Thunderhill box: ~140 ms network, model batches of 14-25 ms; the rest is kev.serve's
event loop and model thread sharing the GIL), more than 16 envs can ask for.

  modal secret create rev-serve-key KEV_API_KEY=$(openssl rand -hex 24)   # once
  REV_SERVE_RUN=rev-0.8b-r6 uvx --from modal==1.5.5 modal deploy training/modal_rev_serve.py
"""

import os
import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modal_rev import HF, RUNS, hf_cache, image, runs  # noqa: E402

SERVE_RUN = os.environ.get("REV_SERVE_RUN", "")
# bf16 (default) or fp32, the exact path the trainer's scoring uses, for precision checks.
SERVE_DTYPE = os.environ.get("REV_SERVE_DTYPE", "bf16")
LABEL = "thunderhill-rev-serve"
WARMUP = {"model": "kev-latest", "state": {"speed km/h": 120, "on track": True},
          "questions": {"q": {"type": "noul", "instructions": "Is the motorcycle on the track?"}}}

app = modal.App("thunderhill-rev-serve")


@app.cls(image=image.env({"REV_SERVE_RUN": SERVE_RUN, "REV_SERVE_DTYPE": SERVE_DTYPE}).add_local_python_source("modal_rev"), gpu="H100", cpu=4, memory=(16384, 65536),
         volumes={RUNS: runs, HF: hf_cache}, secrets=[modal.Secret.from_name("rev-serve-key")],
         min_containers=1, max_containers=1, scaledown_window=600, timeout=3600, startup_timeout=900)
@modal.concurrent(max_inputs=64)
class Serve:
    @modal.enter()
    def load(self):
        import torch
        from kev.api import SystemOneRequest
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server
        from kev.serve import app as api

        run = os.environ["REV_SERVE_RUN"]
        checkpoint = Path(RUNS) / run / "checkpoint"
        if not (checkpoint / "head.pt").exists():
            raise FileNotFoundError(f"no Rev checkpoint at /runs/{run}")
        started = time.time()
        ck = Checkpoint(str(checkpoint))
        bf16 = os.environ["REV_SERVE_DTYPE"] == "bf16"  # the fused kernels are bf16 only
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16 if bf16 else torch.float32,
                                                 cuda_graphs=True, fused=bf16))
        server = api.state.server = Server(ck, tok, model, "cuda")
        server.answer(SystemOneRequest.model_validate(WARMUP))
        server.wait_idle()
        hf_cache.commit()
        print(f"serving {run} on {torch.cuda.get_device_name(0)}, ready in {time.time() - started:.0f}s", flush=True)
        self.api = api

    @modal.asgi_app(label=LABEL)
    def web(self):
        return self.api
