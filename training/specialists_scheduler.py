"""One queue of specialist circuits over every host that trains them: up to
--modal-lanes Modal lanes (modal_specialists.py; the account runs 10 machines at
once), one lane on this machine (morph) and one on skeptrune-thelio. Each lane
runs sac_specialists.py on one circuit with the same stop rule and apex bar.

  systemd-run --user --collect --unit thunderhill-specialists-scheduler \\
      -p Restart=on-failure -p RestartSec=30 --working-directory=$PWD \\
      --setenv=PATH="$PATH" --setenv=HOME="$HOME" \\
      ~/.local/bin/uvx --from modal==1.5.5 python training/specialists_scheduler.py

The scheduler is the only dispatcher, and its state file (--state) holds a claim
per running circuit, so no circuit runs on two hosts. A circuit's run stays on
the host that has it (its home: the Modal volume, this checkout or thelio's);
a circuit without a run goes to the first free host, local hosts first. Every
--poll seconds it:

  - checks each claim: a Modal lane by its function call (or, for a lane it
    adopted without one, by its lane registry and how recently its train.log
    changed), a local lane by its systemd unit;
  - on exit, brings the result here and publishes it to runs/sac/specialists.json
    if it meets the bar measured here (Modal: modal_specialists.sync_track;
    thelio: rsync, then the same check; morph: the lane publishes itself). A
    circuit that meets the bar, or whose run says stuck, is finished; anything
    else (a failure, a stop, a result the check here rejects) is queued again at
    the front, for its home;
  - starts queued circuits on free hosts: morph only with --morph-min-free-gb
    free, Modal only while the estimated spend is under --modal-budget-usd.

Modal spend is estimated as lane wall time times --lane-usd-per-hour.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import modal_specialists as lanes  # noqa: E402
from sac_specialists import publish_checked  # noqa: E402

REGISTRY = ROOT / "runs/sac/specialists.json"
GODOT = "~/.local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64"
THELIO = ["ssh", "-i", str(Path.home() / ".ssh/arguflow"), "-o", "ConnectTimeout=20",
          "skeptrune@skeptrune-thelio"]
THELIO_ROOT = "git_projects/morph/thunderhill-rl"
THELIO_RUNS = "runs/sac/specialists-thelio"
LOCAL_RUNS = "runs/sac/specialists"


def log(message: str):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.state = (json.loads(args.state.read_text()) if args.state.exists()
                      else dict(claims={}, finished={}, homes={}, queue=args.queue,
                                modal_spend_usd=args.modal_spend_before_usd))
        self.lane = modal.Function.from_name(lanes.app.name, "lane")

    def save(self):
        tmp = self.args.state.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1) + "\n")
        tmp.replace(self.args.state)

    # Hosts ----------------------------------------------------------------------
    def running(self, host: str) -> list[str]:
        return [t for t, claim in self.state["claims"].items() if claim["host"] == host]

    def free(self, host: str) -> bool:
        if host == "modal":
            return (len(self.running("modal")) < self.args.modal_lanes
                    and self.state["modal_spend_usd"] < self.args.modal_budget_usd)
        if host == "morph":
            return (not self.running("morph")
                    and shutil.disk_usage(ROOT).free / 1e9 >= self.args.morph_min_free_gb)
        return not self.running(host)

    def start(self, track: str, host: str):
        unit = f"thunderhill-specialist-{track}"
        common = ["--tracks", track, "--reuse", "", "--no-wandb"]
        if host == "modal":
            handle = self.lane.spawn(track).object_id
        elif host == "morph":
            subprocess.run(
                ["systemd-run", "--user", "--collect", "--unit", unit, "-p", "Restart=on-failure",
                 "-p", "RestartSec=30", "-p", "StartLimitIntervalSec=0",
                 f"--working-directory={ROOT}", f"--setenv=PATH={Path.home()}/.local/bin:/usr/bin:/bin",
                 f"--setenv=HOME={Path.home()}", str(Path.home() / ".local/bin/uv"), "run", "--script",
                 "training/sac_specialists.py", "--godot", str(Path(GODOT).expanduser()), *common],
                check=True)
            handle = unit
        else:
            remote = (f"cd {THELIO_ROOT} && git pull -q && systemd-run --user --collect --unit {unit} "
                      "-p Restart=on-failure -p RestartSec=30 -p StartLimitIntervalSec=0 "
                      '--working-directory=$PWD --setenv=PATH="$HOME/.local/bin:/usr/bin:/bin" '
                      '--setenv=HOME="$HOME" $HOME/.local/bin/uv run --script '
                      f"training/sac_specialists.py --godot {GODOT} {' '.join(common[:2])} "
                      f"--reuse '' --no-wandb --runs-dir {THELIO_RUNS} "
                      f"--registry {THELIO_RUNS}/{track}.json --workers 24")
            subprocess.run([*THELIO, remote], check=True)
            handle = unit
        self.state["claims"][track] = dict(host=host, handle=handle, since=time.time(),
                                           accounted=time.time())
        self.state["homes"][track] = host
        log(f"START {track} on {host} ({handle})")

    def alive(self, track: str, claim: dict) -> bool:
        if claim["host"] == "modal":
            if claim["handle"] is None:  # adopted: no function call to ask
                if lanes._read(f"{lanes.LANES}/{track}.json") is not None:
                    return False
                # train.log stays open, so the volume never shows it change; the
                # checkpoints, apex.jsonl and focus.json of every evaluation do.
                newest = max(e.mtime for e in lanes.volume.listdir(
                    f"{lanes.LANES.removeprefix('runs/')}/specialist-{track}"))
                return time.time() - newest < self.args.stale_s
            try:
                modal.FunctionCall.from_id(claim["handle"]).get(timeout=0)
                return False
            except TimeoutError:
                return True
            except Exception as error:  # the lane failed after its retries
                log(f"FAILED {track} on modal: {type(error).__name__}: {str(error)[:300]}")
                return False
        command = ["systemctl", "--user", "is-active", claim["handle"]]
        if claim["host"] == "thelio":
            command = [*THELIO, " ".join(command)]
        state = subprocess.run(command, capture_output=True, text=True).stdout.strip()
        return state in ("active", "activating", "reloading")

    # Results ---------------------------------------------------------------------
    def harvest(self, track: str, host: str) -> str:
        """Bring an exited lane's result here: finished (done or stuck) or requeue."""
        if host == "modal":
            outcome = lanes.sync_track(track)
            phase = json.loads(lanes._read(f"{lanes.LANES}/specialist-{track}/specialist.json")
                               or b'{"phase": "learn"}')["phase"]
            if outcome == "fails-check":
                self.reopen_modal(track)
                phase = "refine"
        elif host == "thelio":
            local = ROOT / THELIO_RUNS
            local.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--exclude", "replay_buffer.npz", "--exclude", "workers",
                            "--exclude", "wandb", "-e", " ".join(THELIO[:-1]),
                            f"{THELIO[-1]}:{THELIO_ROOT}/{THELIO_RUNS}/", f"{local}/"], check=True)
            lane_registry = local / f"{track}.json"
            outcome = "unpublished"
            if lane_registry.exists():
                published, entry = publish_checked(
                    REGISTRY, track, local / f"specialist-{track}",
                    json.loads(lane_registry.read_text())[track], 6, "thelio")
                outcome = "published" if published else "fails-check"
                if not published:
                    log(f"CHECK {track}: fails here: {json.dumps(entry)}")
                    self.reopen_thelio(track)
                    lane_registry.unlink()
            phase = json.loads((local / f"specialist-{track}/specialist.json").read_text()
                               if (local / f"specialist-{track}/specialist.json").exists()
                               else '{"phase": "learn"}')["phase"]
        else:
            registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
            outcome = "published" if registry.get(track, {}).get("meets_bar") else "unpublished"
            path = ROOT / LOCAL_RUNS / f"specialist-{track}/specialist.json"
            phase = json.loads(path.read_text())["phase"] if path.exists() else "learn"
        if outcome == "published":
            return "done"
        if phase == "stuck":
            return "stuck"
        return "requeue"

    def reopen_modal(self, track: str):
        """A lane published a result the check here rejects (e.g. judged by an
        older apex_report): drop its lane registry entry and put its run back in
        refine, so the relaunched lane trains on instead of republishing."""
        run = f"{lanes.LANES.removeprefix('runs/')}/specialist-{track}"
        state = json.loads(lanes._read(f"{lanes.LANES}/specialist-{track}/specialist.json"))
        state.update(phase="refine", reopened=time.time())
        with lanes.volume.batch_upload(force=True) as batch:
            batch.put_file(io.BytesIO((json.dumps(state, indent=1) + "\n").encode()),
                           f"{run}/specialist.json")
        lanes.volume.remove_file(f"{lanes.LANES.removeprefix('runs/')}/{track}.json")
        log(f"REOPEN {track} on modal: lane registry removed, phase refine")

    def reopen_thelio(self, track: str):
        run = f"{THELIO_ROOT}/{THELIO_RUNS}"
        script = (f"import json; p='{run}/specialist-{track}/specialist.json'; "
                  "s=json.load(open(p)); s['phase']='refine'; json.dump(s, open(p, 'w'), indent=1)")
        subprocess.run([*THELIO, f'python3 -c "{script}" && rm -f {run}/{track}.json'], check=True)
        log(f"REOPEN {track} on thelio: lane registry removed, phase refine")

    # Loop ------------------------------------------------------------------------
    def step(self):
        now = time.time()
        for track, claim in list(self.state["claims"].items()):
            if claim["host"] == "modal":
                self.state["modal_spend_usd"] += (now - claim["accounted"]) / 3600 \
                    * self.args.lane_usd_per_hour
                claim["accounted"] = now
            if self.alive(track, claim):
                continue
            del self.state["claims"][track]
            result = self.harvest(track, claim["host"])
            hours = (now - claim["since"]) / 3600
            log(f"EXIT {track} on {claim['host']} after {hours:.2f} h: {result}")
            if result == "requeue":
                self.state["queue"].insert(0, track)
            else:
                self.state["finished"][track] = result
            self.save()
        for track in list(self.state["queue"]):
            home = self.state["homes"].get(track)
            hosts = [home] if home else ["thelio", "morph", "modal"]
            host = next((h for h in hosts if self.free(h)), None)
            if host is None:
                continue
            self.state["queue"].remove(track)
            self.start(track, host)
            self.save()
        self.save()

    def run(self):
        log(f"claims {self.state['claims']}, queue {self.state['queue']}, "
            f"finished {self.state['finished']}, modal spend ${self.state['modal_spend_usd']:.2f}")
        while True:
            try:
                self.step()
            except Exception as error:  # a transient ssh or Modal error: retry next poll
                log(f"ERROR {type(error).__name__}: {str(error)[:500]}")
            if not self.state["claims"] and not self.state["queue"]:
                log(f"ALL FINISHED: {self.state['finished']}")
                return
            time.sleep(self.args.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", type=Path, default=ROOT / "runs/sac/specialists-scheduler.json")
    parser.add_argument("--queue", default="brno,aragon,assen,valencia,cota,goiania,misano,"
                        "buriram,balaton-park,red-bull-ring", help="initial queue (new state only)")
    parser.add_argument("--adopt", default="", help="new state only: TRACK=HOST[:HANDLE] claims "
                        "for lanes already running; a Modal lane without a handle is tracked "
                        "by its lane registry and its run dir's newest change")
    parser.add_argument("--homes", default="", help="new state only: TRACK=HOST where a "
                        "circuit's run already lives")
    parser.add_argument("--modal-lanes", type=int, default=10)
    parser.add_argument("--modal-budget-usd", type=float, default=250.0)
    parser.add_argument("--modal-spend-before-usd", type=float, default=0.0,
                        help="new state only: Modal spend before the scheduler")
    parser.add_argument("--lane-usd-per-hour", type=float, default=1.60,
                        help="16 cores $0.754 + 32 GiB $0.256 + T4 $0.590 (modal.com/pricing)")
    parser.add_argument("--morph-min-free-gb", type=float, default=30.0)
    parser.add_argument("--stale-s", type=float, default=1200.0)
    parser.add_argument("--poll", type=float, default=60.0)
    args = parser.parse_args()
    args.queue = [t for t in args.queue.split(",") if t]
    fresh = not args.state.exists()
    scheduler = Scheduler(args)
    if fresh:
        for item in filter(None, args.adopt.split(",")):
            track, where = item.split("=")
            host, _, handle = where.partition(":")
            since = time.time()
            scheduler.state["claims"][track] = dict(host=host, handle=handle or None, since=since,
                                                    accounted=since)
            scheduler.state["homes"][track] = host
        for item in filter(None, args.homes.split(",")):
            track, host = item.split("=")
            scheduler.state["homes"][track] = host
        scheduler.state["queue"] = [t for t in scheduler.state["queue"]
                                    if t not in scheduler.state["claims"]]
        scheduler.save()
    scheduler.run()


if __name__ == "__main__":
    main()
