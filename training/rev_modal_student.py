# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "gymnasium>=1.1"]
# ///
"""A larger Rev on Modal, distilled from the same data as the local DAgger student.

rev_dagger.py (student A, Rev-0.8B served locally) writes a new aggregated dataset every
round, runs/rev/data-TAG<n>: the base SAC records and every DAgger collection so far,
each state labelled by its circuit's current teacher. This loop trains a larger Rev on
the newest one each time a new one lands:

  1. train: modal_rev.py --size SIZE for one epoch, the first run from the plain Qwen base
     at the size's recipe learning rate, each later one continuing from this loop's previous
     run (--init-from, our own Rev only) at --lr on the newer data, so the epochs add up
     across rounds (one epoch of ~80k records on Rev-4B is hours; two would near the
     12-hour Modal timeout);
  2. serve: modal_rev_serve.py deploys the run on one H100, once --min-free-slots of Rev's
     machine-wide Godot slots are free (rev_drive.GodotSlots), so the endpoint is not
     billed while student A holds the simulators;
  3. evaluate: rev_drive.py rides every circuit and the held-out test tracks through the
     endpoint, all rides at once (one request's round trip is ~0.5 s, so throughput comes
     from concurrency);
  4. stop: `modal app stop thunderhill-rev-serve`, always, also after a failure.
Each evaluation goes to --out/rounds.jsonl in rev_dagger's per-circuit format, with the
Modal cost of the training and serving apps over the run.

Resumable like rev_dagger: a finished training or evaluation is skipped. No new training
starts after --until.

  uv run training/rev_modal_student.py --godot GODOT --size 4b --tag m
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_dagger import ROOT, better, done, evaluation, run  # noqa: E402

MODAL = ["uvx", "--from", "modal==1.5.5", "modal"]
SERVE_APP = "thunderhill-rev-serve"
KEY = Path.home() / ".config/thunderhill-rev/serve_key"


def newest_data(tag: str) -> Path | None:
    finished = [p.parent for p in (ROOT / "runs/rev").glob(f"data-{tag}*/summary.json")
                if re.fullmatch(rf"data-{tag}\d+", p.parent.name)]
    return max(finished, key=lambda p: int(p.name.removeprefix(f"data-{tag}")), default=None)


def free_slots() -> int:
    from rev_drive import GodotSlots
    return GodotSlots().free()


def modal_cost(since: float) -> dict:
    """Modal's billed cost per app (thunderhill-rev*, whole hours) since a local time."""
    start = time.strftime("%Y-%m-%dT%H:00:00", time.localtime(since))
    out = subprocess.run([*MODAL, "billing", "report", "--start", start, "-r", "h", "--tz", "local", "--json"],
                         capture_output=True, text=True, check=True).stdout
    cost = {}
    for row in json.loads(out):
        if row["description"].startswith("thunderhill-rev"):
            cost[row["description"]] = round(cost.get(row["description"], 0.0) + float(row["cost"]), 2)
    return cost


class Endpoint:
    """modal_rev_serve.py deployed for one run for the duration of a with block."""

    def __init__(self, name: str, log: Path):
        self.name, self.log = name, log

    def __enter__(self):
        env = {**os.environ, "REV_SERVE_RUN": self.name}
        with self.log.open("a") as stream:
            deployed = subprocess.run([*MODAL, "deploy", "training/modal_rev_serve.py"], cwd=ROOT, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            stream.write(deployed.stdout)
        if deployed.returncode:
            self.stop()
            raise RuntimeError(f"modal deploy failed; see {self.log}")
        self.url = re.search(r"https://\S+--thunderhill-rev-serve\.modal\.run", deployed.stdout).group(0)
        request = urllib.request.Request(f"{self.url}/v1/models",
                                         headers={"authorization": f"Bearer {KEY.read_text().strip()}"})
        for _ in range(90):  # the container loads the checkpoint and captures its warm-up graphs
            try:
                card = json.load(urllib.request.urlopen(request, timeout=30))["models"][0]
                print(f"[{time.strftime('%H:%M:%S')}] serving {self.name} at {self.url}: {card['run']}", flush=True)
                return self
            except OSError:
                time.sleep(10)
        self.stop()
        raise TimeoutError(f"{self.url} did not come up")

    def stop(self):
        subprocess.run([*MODAL, "app", "stop", "-y", SERVE_APP], cwd=ROOT, capture_output=True)

    def __exit__(self, *exc):
        self.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--godot", required=True)
    parser.add_argument("--size", default="4b")
    parser.add_argument("--tag", default="m", help="student A's round tag: its data is runs/rev/data-TAG<n>")
    parser.add_argument("--lr", type=float, default=2.5e-5, help="continuing runs (the first uses the recipe's)")
    parser.add_argument("--tracks", default="all")
    parser.add_argument("--heldout-tracks", default="portimao,laguna-seca")
    parser.add_argument("--eval-starts", type=int, default=1)
    parser.add_argument("--heldout-starts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=26, help="concurrent rides; Godot slots cap the envs")
    parser.add_argument("--min-free-slots", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/rev/modal-student")
    parser.add_argument("--until", help="local time (YYYY-MM-DDTHH:MM) after which no new training starts")
    parser.add_argument("--poll", type=float, default=300.0)
    args = parser.parse_args()
    deadline = time.mktime(time.strptime(args.until, "%Y-%m-%dT%H:%M")) if args.until else None
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "rev_modal_student.log"
    os.environ["REV_API_KEY"] = KEY.read_text().strip()
    runs_dir = ROOT / "runs/rev"
    state_path, best_path = args.out / "state.json", args.out / "best.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": []}
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    while True:
        data = newest_data(args.tag)
        trained = {entry["data"] for entry in state["runs"]}
        if data is not None and str(data.relative_to(ROOT)) not in trained:
            if deadline is not None and time.time() > deadline:
                print("REV MODAL STUDENT: past --until, no new training", flush=True)
                break
            previous = state["runs"][-1]["name"] if state["runs"] else None
            name = f"rev-{args.size}-{data.name.removeprefix('data-')}"
            began = time.time()
            if not done(runs_dir / name, "result.json"):
                run([*MODAL, "run", "training/modal_rev.py::train", "--data", data, "--name", name, "--size", args.size,
                     "--epochs", 1, *(["--init-from", previous, "--lr", args.lr] if previous else [])], log)
            state["runs"].append({"name": name, "data": str(data.relative_to(ROOT)), "init_from": previous,
                                  "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "began": began})
            state_path.write_text(json.dumps(state, indent=1) + "\n")
        pending = [entry for entry in state["runs"] if "evaluated_at" not in entry]
        for entry in pending:
            ride = f"{entry['name']}-eval"
            if not done(runs_dir / ride, "summary.json"):
                while free_slots() < args.min_free_slots:
                    time.sleep(60)
                with Endpoint(entry["name"], log) as endpoint:
                    run(["uv", "run", "training/rev_drive.py", "--godot", args.godot, "--run-name", ride,
                         "--rev-url", endpoint.url, "--tracks", args.tracks, "--starts", args.eval_starts,
                         "--heldout-tracks", args.heldout_tracks, "--heldout-starts", args.heldout_starts,
                         "--workers", args.workers], log)
            result = {"rev": entry["name"], "data": entry["data"], "init_from": entry["init_from"],
                      **evaluation(runs_dir / ride), "ride": f"runs/rev/{ride}",
                      "modal_cost_usd": modal_cost(entry["began"]), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with (args.out / "rounds.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            if better(result, best):
                best = result
                best_path.write_text(json.dumps(best, indent=1) + "\n")
            entry["evaluated_at"] = result["time"]
            state_path.write_text(json.dumps(state, indent=1) + "\n")
            print(f"REV MODAL STUDENT {entry['name']}: {result['laps']}/{result['rides']} laps on "
                  f"{result['circuits_lapped']} circuits, held out {result['heldout_laps']} laps, "
                  f"cost {result['modal_cost_usd']} | best {best['rev']} {best['laps']} laps", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
