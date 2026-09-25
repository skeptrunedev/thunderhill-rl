"""Exercise native rollout dispatch ordering on CUDA, without a model or training.

Executes the exact pinned and patched rollout_generation method bodies with a
minimal session worker to isolate the cross-thread weight-copy dependency.
"""

import argparse
import ast
import importlib.util
import json
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
