# /// script
# requires-python = ">=3.11"
# dependencies = ["Pillow>=11,<13"]
# ///
"""Verify real rendered camera observations and capture/advance serialization."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from PIL import Image, ImageChops

from check_agent import Client
from remote_godot import sync_if_remote

ROOT = Path(__file__).resolve().parents[1]


# This tests observation stability, not GPU bit identity. On the RTX PRO 6000
# archived preflight, 0.346% of pixels varied by at most 3 byte levels while
# pose and ticks were identical (mean channel difference 0.001623).
# Godot 4.7 mobile/tonemap dither is spatial, not temporal; background shader
# specialization is a possible cause, not established by this measurement.
# https://docs.godotengine.org/en/4.5/tutorials/performance/pipeline_compilations.html
# The maximum is a 2% full-scale channel tolerance across GPU backends, not
# a device-specific observed maximum. Mean drift and affected area remain much
# stricter. Previously saved receipts still require exact byte/hash immutability.
# The F-theta lens pass resamples each view bilinearly, which spreads that
# jitter over more pixels while shrinking it: an H100 idle rerender changed
# 0.56% of pixels by at most one level. A one-level flip is sub-LSB rounding,
# so the affected-area limit counts only changes above one level.
IDLE_IMAGE_LIMITS = dict(max_channel_change=255 * 0.02, mean_channel_change=0.005,
                         changed_pixel_fraction=0.005)
IDLE_ROUNDING_LEVELS = 1


def validate_idle_capture(first_image, repeated_image, first_camera, repeated_camera):
    if first_camera != repeated_camera:
        raise AssertionError("Idle capture changed camera calibration or pose")
    if first_image.size != repeated_image.size or first_image.mode != repeated_image.mode:
        raise AssertionError("Idle capture changed image dimensions or format")
    difference = ImageChops.difference(first_image, repeated_image)
    pixels = list(difference.getdata())
    channels = len(difference.getbands())
    stats = dict(max_channel_change=max(high for low, high in difference.getextrema()),
                 mean_channel_change=sum(sum(pixel) for pixel in pixels) / (len(pixels) * channels),
                 changed_pixel_fraction=sum(max(pixel) > IDLE_ROUNDING_LEVELS for pixel in pixels)
                 / len(pixels))
    if any(stats[key] > limit for key, limit in IDLE_IMAGE_LIMITS.items()):
        raise AssertionError(f"Idle camera image is unstable: {stats}; limits={IDLE_IMAGE_LIMITS}")
    return stats


@contextmanager
def game(args, output: Path, headless: bool = False):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ, DISPLAY=args.display, XDG_DATA_HOME=str(output / "userdata"),
               THUNDERHILL_CAPTURE_DIAGNOSTICS="1" if args.trace_render else "0")
    command = [args.godot, "--path", str(ROOT / "godot"), "--audio-driver", "Dummy"]
    if headless:
        command.append("--headless")
    else:
        driver = "opengl3" if args.rendering_method == "gl_compatibility" else "vulkan"
        command += ["--resolution", "800x500", "--position", "40,40",
                    "--rendering-method", args.rendering_method, "--rendering-driver", driver]
    command += ["--", f"--agent-port={port}"]
    if args.offscreen and not headless:
        command.append("--agent-offscreen")
    log_path = output / ("headless.log" if headless else "rendered.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        connection = None
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(log_path.read_text())
                try:
                    connection = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                    break
                except OSError:
                    time.sleep(0.1)
            if connection is None:
                raise TimeoutError(log_path.read_text())
            connection.settimeout(30)
            yield Client(connection)
        except Exception:
            print(log_path.read_text())
            raise
        finally:
            if connection:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def validate_capture(response: dict, output: Path, label: str, episode: str, tick: int, *, view_only=False) -> dict:
    assert "error" not in response, response
    assert response["episode_id"] == episode and response["tick"] == tick, response
    assert not {"state", "track", "transitions", "reward_components"}.intersection(response), response.keys()
    image = response["image"]
    png = base64.b64decode(image["base64"], validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    digest = hashlib.sha256(png).hexdigest()
    assert digest == image["sha256"]
    decoded = Image.open(io.BytesIO(png))
    assert decoded.size == (image["width"], image["height"]) == (512, 320)
    assert decoded.convert("RGB").entropy() > 3.0, "Observation appears blank"
    assert response["camera"]["hud_visible"] is False
    assert response["camera"]["rider_mesh_visible"] is False
    assert response["camera"]["bike_mesh_visible"] is False
    intrinsics = response["camera"]["intrinsics"]
    assert intrinsics["model"] == "ftheta" and (intrinsics["width"], intrinsics["height"]) == (512, 320)
    assert 0 < intrinsics["cx"] < 512 and 0 < intrinsics["cy"] < 320
    rig = response["camera"]["rig"]
    assert rig["up"] == [0.0, 1.0, 0.0] and abs(rig["forward"][1]) < 1e-9, "Policy rig must be level"
    if not view_only:
        views = response["views"]
        assert [view["logical_id"] for view in views] == [
            "camera_cross_left_120fov", "camera_front_wide_120fov",
            "camera_cross_right_120fov", "camera_front_tele_30fov"]
        assert len({view["image"]["sha256"] for view in views}) == 4, "Duplicated camera images"
        for view in views:
            validate_capture(view, output, label + "_" + view["logical_id"], episode, tick, view_only=True)
        assert views[1]["image"] == response["image"]
        assert views[0]["camera"]["pose"] != views[2]["camera"]["pose"]
        assert views[1]["camera"]["rig"] == views[3]["camera"]["rig"]
    assert len(response["camera"]["pose"]["position"]) == 3
    sync_if_remote(output / "userdata")
    artifacts = list((output / "userdata").rglob(digest + ".png"))
    assert len(artifacts) == 1 and artifacts[0].read_bytes() == png, "Recorded bytes differ from policy image"
    (output / f"{label}.png").write_bytes(png)
    metadata = {k: v for k, v in response.items() if k != "views"}
    metadata["image"] = {k: v for k, v in image.items() if k != "base64"}
    if not view_only:
        metadata["views"] = [
            {**view, "image": {k: v for k, v in view["image"].items() if k != "base64"}}
            for view in response["views"]]
    (output / f"{label}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"sha256": digest, "mtime_ns": artifacts[0].stat().st_mtime_ns}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", default=shutil.which("godot") or shutil.which("godot4"))
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rendering-method", choices=["gl_compatibility", "mobile", "forward_plus"],
                        default="gl_compatibility")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--require-hardware", action="store_true")
    parser.add_argument("--initial-capture-timeout", type=float, default=30)
    parser.add_argument("--trace-render", action="store_true")
    args = parser.parse_args()
    if not args.godot:
        parser.error("Provide --godot")
    output = args.output or ROOT / "artifacts/qa" / datetime.now(timezone.utc).strftime("camera-%Y%m%dT%H%M%SZ")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with game(args, output) as client:
        initial = client.request({"op": "reset", "policy_id": "camera-qa"})
        episode = initial["episode_id"]
        capture = {"op": "capture", "episode_id": episode, "expected_tick": 0}
        client.connection.settimeout(args.initial_capture_timeout)
        first = client.request(capture)
        client.connection.settimeout(30)
        first_info = validate_capture(first, output, "tick0", episode, 0)
        assert first["camera"]["renderer"]["offscreen"] is args.offscreen
        assert first["camera"]["renderer"]["method"] == args.rendering_method
        if args.require_hardware:
            adapter = first["camera"]["renderer"]["adapter"].lower()
            assert adapter and not any(name in adapter for name in
                ("llvmpipe", "lavapipe", "swiftshader", "software")), first["camera"]["renderer"]
        time.sleep(0.25)
        assert client.request({"op": "observe", "episode_id": episode}) == initial, "Rendering advanced physics"
        repeated = client.request(capture)
        repeat_info = validate_capture(repeated, output, "tick0_repeat", episode, 0)
        assert repeated["camera"] == first["camera"], "Idle capture changed camera calibration or pose"
        if repeat_info["sha256"] == first_info["sha256"]:
            assert repeat_info["mtime_ns"] == first_info["mtime_ns"], "Identical artifact was overwritten"
        idle_difference = ImageChops.difference(Image.open(output / "tick0.png"), Image.open(output / "tick0_repeat.png"))
        idle_max_channel_change = max(high for low, high in idle_difference.getextrema())
        idle_stability = None
        if args.offscreen:
            for first_view, repeated_view in zip(first["views"], repeated["views"], strict=True):
                validate_idle_capture(
                    Image.open(io.BytesIO(base64.b64decode(first_view["image"]["base64"]))),
                    Image.open(io.BytesIO(base64.b64decode(repeated_view["image"]["base64"]))),
                    first_view["camera"], repeated_view["camera"])
            idle_stability = validate_idle_capture(
                Image.open(output / "tick0.png"), Image.open(output / "tick0_repeat.png"),
                first["camera"], repeated["camera"])
        advance = {"op": "advance", "episode_id": episode, "expected_tick": 0,
                   "action_id": "camera-first", "controls": {"throttle": 0.8}}
        advanced = client.request(advance)
        assert advanced["tick"] == 12
        capture["expected_tick"] = 12
        second = client.request(capture)
        second_info = validate_capture(second, output, "tick12", episode, 12)
        assert second_info["sha256"] != first_info["sha256"], "Movement did not change observation"
        # Queue both commands before reading. Advance must wait for actual frame capture.
        queued_advance = {**advance, "expected_tick": 12, "action_id": "camera-queued"}
        client.stream.write((json.dumps(capture) + "\n" + json.dumps(queued_advance) + "\n").encode())
        client.stream.flush()
        captured_before_advance = json.loads(client.stream.readline())
        queued_result = json.loads(client.stream.readline())
        queued_info = validate_capture(captured_before_advance, output, "tick12_queued", episode, 12)
        assert captured_before_advance["camera"] == second["camera"], "Queued advance contaminated capture pose"
        if queued_info["sha256"] == second_info["sha256"]:
            assert queued_info["mtime_ns"] == second_info["mtime_ns"], "Immutable capture overwritten"
        assert queued_result["tick"] == 24, queued_result
        # Reset is another state mutation and must also wait for capture completion.
        capture["expected_tick"] = 24
        client.stream.write((json.dumps(capture) + "\n" + json.dumps({"op": "reset", "policy_id": "after-camera"}) + "\n").encode())
        client.stream.flush()
        before_reset = json.loads(client.stream.readline())
        reset = json.loads(client.stream.readline())
        validate_capture(before_reset, output, "tick24_queued_reset", episode, 24)
        assert reset["tick"] == 0 and reset["episode_id"] != episode
        assert "error" in client.request(capture), "Stale capture episode was accepted"
    with game(args, output, headless=True) as client:
        initial = client.request({"op": "reset"})
        unsupported = client.request({"op": "capture", "episode_id": initial["episode_id"], "expected_tick": 0})
        assert "headless" in unsupported["error"] and unsupported["failure_type"] == "infrastructure"
    records = []
    sync_if_remote(output / "userdata")
    for path in (output / "userdata").rglob("*.jsonl"):
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    observations = [row for row in records if row.get("type") == "camera_observation"]
    assert len(observations) == 5, len(observations)
    assert all("base64" not in row["image"] and all("base64" not in view["image"] for view in row["views"]) for row in observations)
    summary = {"ok": True, "renderer": first["camera"]["renderer"], "camera_observations_recorded": len(observations), "dimensions": [512, 320], "views_per_observation": 4,
               "idle_rerender_max_channel_change": idle_max_channel_change,
               "idle_image_stability": idle_stability, "idle_image_limits": IDLE_IMAGE_LIMITS,
               "checks": ["four_distinct_calibrated_views_same_tick", "real_png", "sha256", "immutable_artifact", "deterministic_idle_pose",
                          "no_privileged_telemetry", "frozen_tick", "changed_pixels_after_advance",
                          "queued_advance_serialization", "queued_reset_serialization", "headless_rejection"],
               "output": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
