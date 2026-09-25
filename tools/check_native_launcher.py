"""Qualify the reviewed native process monitor with real CPU child processes.

This executes the exact native monitor definition without importing CUDA model
packages. The running children are real processes, including a deliberate worker
exit and a descendant in its own session, cleaned by our actual run supervisor.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

from training.native_cosmos_source import REVISION, verify_launcher
from training.nvidia_alpagym import owned_subreaper, stop_process_tree, wait_process


def monitor(root: Path, scenario: str, receipt: Path) -> None:
    source = root / "cosmos_rl/launcher/launch_all.py"
    verify_launcher(root)
    definition = next(
        node for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "wait_for_processes"
    )
    namespace = {"logger": logging.getLogger("native-monitor"), "sys": sys, "time": time}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    children = []
    try:
        for index in range(3):
            delay, code = (0.2, 7) if index == 2 and scenario == "worker-failure" else (0.5, 0)
            if scenario == "worker-failure" and index != 2:
                delay = 60
            children.append(subprocess.Popen([
                sys.executable, "-c",
                "import time,sys; time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))",
                str(delay), str(code),
            ]))
        # The real outer subreaper must also contain children in other sessions.
        detached = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        receipt.write_text(json.dumps({"pids": [child.pid for child in children] + [detached.pid]}))
        namespace["wait_for_processes"](
            children, controller_process=children[0], fail_fast_workers=True
        )
    finally:
        for child in children:
            if child.poll() is not None:
                child.wait()


def qualify(root: Path, output: Path) -> dict:
    import psutil

    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError("Process qualification requires the pinned Cosmos checkout")
    certificate = verify_launcher(root, apply_patch=True)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for scenario in ("worker-failure", "success"):
        receipt = output / (scenario + ".json")
        began = time.monotonic()
        failure_code = None
        with owned_subreaper():
            process = subprocess.Popen([
                sys.executable, "-m", "tools.check_native_launcher", "--checkout", str(root),
                "--scenario", scenario, "--output", str(receipt),
            ], start_new_session=True)
            try:
                wait_process(process, time.monotonic() + 10)
            except subprocess.CalledProcessError as error:
                failure_code = error.returncode
            finally:
                stop_process_tree(process)
        pids = json.loads(receipt.read_text())["pids"]
        surviving = [pid for pid in pids if psutil.pid_exists(pid)]
        expected = 1 if scenario == "worker-failure" else None
        if failure_code != expected or surviving:
            raise RuntimeError(f"Native monitor failed process qualification: {failure_code}, {surviving}")
        rows.append(dict(scenario=scenario, observed_error_code=failure_code,
                         elapsed_seconds=time.monotonic() - began, surviving_pids=surviving))
    result = dict(passed=True, native_launcher=certificate, scenarios=rows,
                  limitation="CPU process supervision only; no model training or CUDA qualification.")
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=("worker-failure", "success"))
    args = parser.parse_args()
    if args.scenario:
        monitor(args.checkout.resolve(), args.scenario, args.output)
    else:
        print(json.dumps(qualify(args.checkout.resolve(), args.output), indent=2))


if __name__ == "__main__":
    main()
