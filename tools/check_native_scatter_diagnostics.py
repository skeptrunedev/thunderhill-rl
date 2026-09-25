"""Operational CPU/CUDA check of opt-in native scatter receipts, not model training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def child(args):
    import torch
    from alpagym_runtime.scatter_diagnostics import scatter_diagnostics

    os.environ["ALPAGYM_INFERENCE_CAPTURE_DIR"] = str(args.output)
    os.environ["ALPAGYM_SCATTER_DIAGNOSTICS"] = "1"
    target = torch.zeros(4, device=args.device)
    mask = torch.tensor([True, False, True, False], device=args.device)
    values = torch.tensor([7.0, 9.0], device=args.device)
    with scatter_diagnostics():
        if args.invalid:
            try:
                target.masked_scatter(mask, values[:1])
            except (RuntimeError, ValueError):
                receipt = json.loads(
                    (
                        args.output / str(os.getpid()) / "scatter-failure.json"
                    ).read_text()
                )
                assert (
                    receipt["selected_elements"] == 2 and receipt["source_numel"] == 1
                )
                assert receipt["enough_source"] is False
                return
            raise AssertionError("Original malformed primitive did not fail")
        actual = target.masked_scatter(mask, values)
        assert actual.cpu().tolist() == [7.0, 0.0, 9.0, 0.0]
        # Another Python thread must not inherit this thread's dispatcher mode.
        with ThreadPoolExecutor(max_workers=1) as pool:
            other = pool.submit(
                lambda: target.masked_scatter(mask, values).cpu().tolist()
            ).result()
        assert other == [7.0, 0.0, 9.0, 0.0]
        receipt_path = args.output / str(os.getpid()) / "scatter-last-completed.json"
        assert json.loads(receipt_path.read_text())["operator_index"] == 1
        target.masked_scatter_(mask, values)
        receipt = json.loads(receipt_path.read_text())
        assert (
            receipt["operator_index"] == 2 and receipt["stage"] == "operator_completed"
        )
    # Disabled mode leaves no files.
    os.environ["ALPAGYM_SCATTER_DIAGNOSTICS"] = "0"
    before = receipt_path.read_bytes()
    with scatter_diagnostics():
        target.masked_scatter(mask, values)
    assert receipt_path.read_bytes() == before


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--child", action="store_true")
    p.add_argument("--invalid", action="store_true")
    args = p.parse_args()
    if args.child:
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(args.upstream.resolve() / "packages/runtime/src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    rows = []
    for invalid in (False, True):
        output = args.output / ("invalid" if invalid else "valid")
        cmd = [
            sys.executable,
            "-m",
            "tools.check_native_scatter_diagnostics",
            "--upstream",
            str(args.upstream),
            "--output",
            str(output),
            "--device",
            args.device,
            "--child",
        ]
        if invalid:
            cmd.append("--invalid")
        completed = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=60
        )
        (args.output / ("invalid.log" if invalid else "valid.log")).write_text(
            completed.stdout + completed.stderr
        )
        if completed.returncode:
            raise RuntimeError(f"Diagnostic child failed: {completed.stderr}")
        receipts = list(
            output.glob(
                "*/scatter-failure.json" if invalid else "*/scatter-last-completed.json"
            )
        )
        assert len(receipts) == 1
        rows.append(json.loads(receipts[0].read_text()))
    report = {
        "passed": True,
        "device": args.device,
        "thread_scope_verified": True,
        "disabled_mode_verified": True,
        "receipts": rows,
        "limitation": "Synthetic operator check, not model training or root cause reproduction",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "device": args.device,
                "report": str(args.output / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
