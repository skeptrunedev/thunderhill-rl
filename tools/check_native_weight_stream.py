"""Exercise native rollout dispatch ordering on CUDA, without a model or training.

Executes the exact pinned and patched rollout_generation method bodies with a
minimal session worker to isolate the cross-thread weight-copy dependency.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import torch


def load_method(source, fence):
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AlpagymRollout"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "rollout_generation"
    )
    method.decorator_list = []
    namespace = {
        "wait_for_weight_updates": fence,
        "RolloutResult": lambda **kw: kw,
        "logger": SimpleNamespace(info=lambda *a: None),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    # Execute only the inspected native method AST; source provenance is checked
    # by the qualification launcher before this diagnostic is invoked.
    exec(  # noqa: S102
        compile(
            ast.fix_missing_locations(module), "<native rollout_generation>", "exec"
        ),
        namespace,
    )
    return namespace["rollout_generation"]


def check_receive_lifetime(root):
    """Run exact native temp allocation and completion bodies with dtype conversion."""
    source = (root / "cosmos_rl/rollout/worker/rollout_control.py").read_text()
    manifest_path = Path(__file__).parent / "patches/cosmos-receive-lifetime.json"
    manifest = json.loads(manifest_path.read_text())
    expected = manifest["files"]["cosmos_rl/rollout/worker/rollout_control.py"]
    assert hashlib.sha256(source.encode()).hexdigest() == expected["patched_sha256"]
    inserted = """                # Receive storage was allocated on the NCCL stream. This
                # completion runs on a separate copy stream; register its use
                # before enqueueing the read so allocator reuse cannot race it.
                recv_tensor.record_stream(torch.cuda.current_stream(recv_tensor.device))
"""
    original = source.replace(inserted, "")
    assert hashlib.sha256(original.encode()).hexdigest() == expected["original_sha256"]
    rows = []
    for version, body in (("original", original), ("patched", source)):
        tree = ast.parse(body)
        method = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "recv_weight_shard"
        )
        nested = [
            n
            for n in method.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("recv_tensor_creator", "completion_lambda")
        ]
        queue = Queue()
        worker = SimpleNamespace(
            temp_recv_tensor_queue=queue,
            device=torch.device("cuda:0"),
            quantization_type=None,
            parallel_dims=None,
            weight_mapper=SimpleNamespace(
                update_tensor_view=lambda view, recv, name, **kw: view.copy_(recv)
            ),
        )
        namespace = {
            "torch": torch,
            "self": worker,
            "target_dtype": torch.float32,
            "constant": SimpleNamespace(COSMOS_RECV_TENSOR_QUEUE_SIZE=10000),
        }
        # Both method bodies above are authenticated against reviewed pinned hashes.
        exec(  # noqa: S102
            compile(
                ast.Module(body=nested, type_ignores=[]),
                "<native recv functions>",
                "exec",
            ),
            namespace,
        )
        target = torch.zeros((2048, 2048), device="cuda", dtype=torch.bfloat16)
        spare = torch.zeros_like(target)
        torch.cuda.synchronize()  # Diagnostic initialization only.
        receive = torch.cuda.Stream()
        copy = torch.cuda.Stream()
        # Warm the allocator before delaying copies: cudaMalloc itself can
        # synchronize streams and hide the lifetime error under investigation.
        with torch.cuda.stream(receive):
            reserve = [
                torch.empty((2048, 2048), device="cuda", dtype=torch.float32)
                for _ in range(16)
            ]
        receive.synchronize()
        del reserve
        with torch.cuda.stream(receive):
            tensor, event, _ = namespace["recv_tensor_creator"](target)
            tensor.fill_(37)
            pointer = tensor.data_ptr()
            other, _, _ = namespace["recv_tensor_creator"](spare)
            popped_unrecorded = queue.qsize() == 1
            ready = torch.cuda.Event()
            ready.record()
        copy.wait_event(ready)
        with torch.cuda.stream(copy):
            torch.cuda._sleep(1_000_000_000)
            namespace["completion_lambda"]([(target, tensor, event, "fixture")], [], [])
        del tensor
        replacements = []
        with torch.cuda.stream(receive):
            for _ in range(12):
                replacement = torch.empty(
                    (2048, 2048), device="cuda", dtype=torch.float32
                )
                replacement.fill_(91)
                replacements.append(replacement)
        reused = any(t.data_ptr() == pointer for t in replacements)
        copy.synchronize()  # Observe completed diagnostic, not production synchronization.
        receive.synchronize()
        observed = float(target[0, 0].item())
        rows.append(
            {
                "version": version,
                "unrecorded_event_popped": popped_unrecorded,
                "allocation_reused": reused,
                "observed": observed,
            }
        )
        queue.queue.clear()
        del replacements, replacement, other, target, spare
    assert rows[0]["allocation_reused"] and rows[0]["observed"] == 91, rows
    assert not rows[1]["allocation_reused"] and rows[1]["observed"] == 37, rows
    return {
        "passed": True,
        "transfer_dtype": "float32",
        "target_dtype": "bfloat16",
        "trials": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cosmos-source", type=Path)
    args = p.parse_args()
    path = "packages/runtime/src/alpagym_runtime/cosmos/rollout_backend.py"
    spec = importlib.util.spec_from_file_location(
        "handoff",
        args.upstream / "packages/runtime/src/alpagym_runtime/cuda_handoff.py",
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original = subprocess.check_output(
        ["git", "-C", str(args.upstream), "show", "HEAD:" + path], text=True
    )
    methods = {
        "original": load_method(original, helper.wait_for_weight_updates),
        "patched": load_method(
            (args.upstream / path).read_text(), helper.wait_for_weight_updates
        ),
    }
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: torch.cuda.current_stream().cuda_stream).result()
        for name, method in methods.items():
            for trial in range(3):
                weights = torch.zeros(1, device="cuda")
                torch.cuda.synchronize()  # Fixture initialization, outside production handoff.
                copy = torch.cuda.Stream()
                cosmos = torch.cuda.Stream()
                done = torch.cuda.Event()
                with torch.cuda.stream(copy):
                    torch.cuda._sleep(500_000_000)
                    weights.fill_(37)
                    done.record()
                cosmos.wait_event(done)

                def submit(_, weights=weights):
                    future = Future()
                    observed = pool.submit(
                        lambda: float(weights.clone().cpu().item())
                    ).result()
                    future.set_result([observed])
                    return SimpleNamespace(future=future, n_target=1)

                backend = SimpleNamespace(
                    _worker=SimpleNamespace(submit_payload=submit)
                )
                start = time.monotonic()
                result = method(backend, [object()], cosmos, None)
                elapsed = time.monotonic() - start
                done.synchronize()  # Fixture cleanup after result, not a production fence.
                observed = result[0]["completions"][0]
                rows.append(
                    {
                        "version": name,
                        "trial": trial,
                        "observed": observed,
                        "seconds": elapsed,
                    }
                )
    assert all(r["observed"] == 0 for r in rows if r["version"] == "original"), rows
    assert all(r["observed"] == 37 for r in rows if r["version"] == "patched"), rows
    report = {
        "passed": True,
        "device": torch.cuda.get_device_name(),
        "diagnostic": "native method body with synthetic delayed weight copies, not model training",
        "trials": rows,
    }
    if args.cosmos_source is not None:
        report["receive_lifetime"] = check_receive_lifetime(args.cosmos_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
