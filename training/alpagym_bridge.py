"""Godot RuntimeService for NVIDIA's unmodified AlpaGym driver and trainer.

Only the simulator boundary lives here: measured sensors, route geometry,
trajectory execution, metrics and immutable recordings. No model, sampling,
replay objective, optimizer, demonstration or teacher action is implemented.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import uuid
import signal

import grpc
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from alpasim_grpc.v0 import common_pb2 as common
from alpasim_grpc.v0 import egodriver_pb2 as driver
from alpasim_grpc.v0 import egodriver_pb2_grpc as driver_grpc
from alpasim_grpc.v0 import runtime_pb2 as runtime
from alpasim_grpc.v0 import runtime_pb2_grpc as runtime_grpc
from alpasim_grpc.v0 import sensorsim_pb2 as sensor

from agent_harness import ThunderhillEnv
from check_parallel import worker
from driving_trajectory import TrajectoryTracker
from lap_policy import RoadTelemetry
from lap_episode import StallMonitor
from video_jobs import enqueue_video

ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "thunderhill-east-standing"
CAMERA_ID = "camera_front_wide_120fov"
SAMPLE_US = 100_000
WARMUP_TICKS = 180  # 1.5 measured seconds, sixteen poses including t=0
CONTROL_SAMPLES = 2  # NVIDIA example: replan every 0.2 simulated seconds
# Fixed world basis only. Do not rebase each observation onto the latest pose.
GODOT_TO_LOCAL = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def measured_rig(observation, capture):
    """Return position and full measured bank/pitch orientation in AlpaSim axes.

    Camera's fixed downward pitch is removed to recover the rider rig. Keeping
    measured roll avoids pairing a leaned image with invented level egomotion.
    """
    camera = capture["camera"]["pose"]
    forward = -np.asarray(camera["basis_z"], dtype=float)
    camera_up = np.asarray(camera["basis_y"], dtype=float)
    down = 0.08  # agent_camera.gd LOOK_DOWN_RAD
    tangent = forward * math.cos(down) + camera_up * math.sin(down)
    up = -forward * math.sin(down) + camera_up * math.cos(down)
    left = -np.asarray(camera["basis_x"], dtype=float)
    rotation = GODOT_TO_LOCAL @ np.column_stack((tangent, left, up))
    position = GODOT_TO_LOCAL @ np.asarray(
        observation["state"]["position"], dtype=float
    )
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) < 0.999
    ):
        raise ValueError("Camera receipt does not define a right handed rig")
    return position, rotation


def pose_proto(position, rotation):
    q = Rotation.from_matrix(rotation).as_quat()  # scipy xyzw, protobuf wxyz
    return common.Pose(
        vec=common.Vec3(x=position[0], y=position[1], z=position[2]),
        quat=common.Quat(w=q[3], x=q[0], y=q[1], z=q[2]),
    )


def future_in_rig(response, now_us, position, rotation):
    """Upstream returns world poses including t=now; tracker expects future ego."""
    future = [p for p in response.trajectory.poses if p.timestamp_us > now_us]
    if len(future) < CONTROL_SAMPLES:
        raise ValueError("Driver trajectory cannot cover the next control interval")
    times = [p.timestamp_us for p in future]
    if times != [now_us + (i + 1) * SAMPLE_US for i in range(len(times))]:
        raise ValueError("Expected future waypoints on the native 10Hz action grid")
    xyz = np.array([[p.pose.vec.x, p.pose.vec.y, p.pose.vec.z] for p in future])
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite driver trajectory")
    return ((xyz - position) @ rotation).tolist()


def camera_calibration(capture, position, rotation):
    """Truthful calibration from the exact first captured camera receipt."""
    image, camera = capture["image"], capture["camera"]
    intrinsics = camera["intrinsics"]
    spec = sensor.CameraSpec(
        logical_id=CAMERA_ID, resolution_w=image["width"], resolution_h=image["height"]
    )
    pinhole = spec.opencv_pinhole_param
    pinhole.focal_length_x, pinhole.focal_length_y = intrinsics["fx"], intrinsics["fy"]
    pinhole.principal_point_x, pinhole.principal_point_y = (
        intrinsics["cx"],
        intrinsics["cy"],
    )
    pose = camera["pose"]
    # Camera optical axes: X right, Y down, Z forward. Proto field is named
    # rig_to_camera but upstream explicitly documents camera_to_rig semantics.
    optical_world = GODOT_TO_LOCAL @ np.column_stack(
        (pose["basis_x"], -np.asarray(pose["basis_y"]), -np.asarray(pose["basis_z"]))
    )
    offset = (GODOT_TO_LOCAL @ np.asarray(pose["position"]) - position) @ rotation
    return sensor.AvailableCamerasReturn.AvailableCamera(
        logical_id=CAMERA_ID,
        intrinsics=spec,
        rig_to_camera=pose_proto(offset, rotation.T @ optical_world),
    )


class GodotRuntime(runtime_grpc.RuntimeServiceServicer):
    def __init__(self, game, identity_root, *, identity_provider=None):
        self.game = dict(game)
        if Path(game["project_path"]).resolve() != ROOT / "godot":
            raise ValueError("Game project_path must match this checkout")
        seconds = game["episode_seconds"]
        if (
            not math.isfinite(seconds)
            or seconds <= 0
            or abs(round(seconds * 10) - seconds * 10) > 1e-8
        ):
            raise ValueError("Episode length must be positive whole 0.1 second samples")
        if type(game["concurrency"]) is not int or game["concurrency"] < 1:
            raise ValueError("Concurrency must be a positive integer")
        self.slots = threading.BoundedSemaphore(game["concurrency"])
        self.identity_root = Path(identity_root)
        self.identity_provider = identity_provider
        self.road = RoadTelemetry()
        self._counter = 0
        self._lock = threading.Lock()
        self.shutdown = threading.Event()

    def get_runtime_info(self, request, context):
        return runtime.RuntimeInfo(
            max_supported_concurrent_rollouts=self.game["concurrency"],
            nr_workers=self.game["concurrency"],
            renderer_type="godot",
            scenes=[
                runtime.SceneInfo(
                    scene_id=SCENE_ID,
                    provider_kind="godot",
                    metadata=runtime.SceneMetadata(
                        uuid=SCENE_ID,
                        camera_ids=[CAMERA_ID],
                        start_time_us=0,
                        end_time_us=int((self.game["episode_seconds"] + 1.5) * 1e6),
                    ),
                )
            ],
        )

    def shut_down(self, request, context):
        self.shutdown.set()
        return common.Empty()

    def _identity(self, port):
        identity = (
            self.identity_provider(port)
            if self.identity_provider
            else json.loads((self.identity_root / f"driver_{port}.json").read_text())
        )
        if (
            identity.get("active") is not True
            or type(identity.get("policy_version")) is not int
            or identity["policy_version"] < 0
            or not identity.get("batch_id")
            or type(identity.get("is_validation")) is not bool
        ):
            raise ValueError("Missing active upstream policy version provenance")
        return identity

    def simulate(self, request, context):
        if not request.available_drivers or request.n_concurrent_per_driver < 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "A driver and positive concurrency are required",
            )
        jobs = []
        for spec in request.rollout_specs:
            if spec.scenario_id != SCENE_ID or spec.nr_rollouts < 1:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Unknown scene or empty rollout spec",
                )
            if spec.session_uuids and len(spec.session_uuids) != spec.nr_rollouts:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "Session count differs from rollout count",
                )
            for index in range(spec.nr_rollouts):
                session = (
                    spec.session_uuids[index]
                    if spec.session_uuids
                    else uuid.uuid4().hex
                )
                if not session or any(
                    c
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                    for c in session
                ):
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT, "Unsafe session UUID"
                    )
                address = request.available_drivers[
                    len(jobs) % len(request.available_drivers)
                ]
                jobs.append((spec, session, address, self._identity(address.port)))
        if not jobs:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "At least one rollout is required"
            )
        max_workers = min(
            self.game["concurrency"],
            len(request.available_drivers) * request.n_concurrent_per_driver,
        )
        driver_slots = {
            a.port: threading.BoundedSemaphore(request.n_concurrent_per_driver)
            for a in request.available_drivers
        }

        def run_limited(args):
            with driver_slots[args[2].port]:
                return self._run(*args)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(run_limited, jobs))
        return runtime.SimulationReturn(rollout_returns=results)

    def _run(self, spec, session, address, identity):
        with self.slots:
            with self._lock:
                self._counter += 1
                rollout = self._counter
            output = Path(self.game["recording_root"]) / session
            output.mkdir(parents=True, exist_ok=False)
            provenance = dict(
                identity,
                session_uuid=session,
                rollout_number=rollout,
                driver_port=address.port,
                scenario_id=SCENE_ID,
                control_period_seconds=0.2,
                sensor_period_seconds=0.1,
                warmup_seconds=1.5,
                action_semantics="native_trajectory_with_fixed_motorcycle_controller",
            )
            _write_json(output / "provenance.json", provenance)
            result = runtime.SimulationReturn.RolloutReturn(
                rollout_spec=spec, rollout_uuid=session
            )
            try:
                metrics = self._episode(output, session, address, provenance)
                result.success = True  # Protocol success means completed simulation, not a completed lap.
                result.aggregated_metrics.update(metrics)
            except Exception as error:
                result.success = False
                result.error = f"{type(error).__name__}: {error}"
                _write_json(
                    output / "failure.json", dict(error=result.error, **provenance)
                )
            return result

    def _episode(self, output, session, address, provenance):
        display = dict(
            model_name="nvidia/Alpamayo-1.5-10B",
            generation=provenance["policy_version"],
        )
        if provenance["is_validation"]:
            display["evaluation"] = True
        else:
            display["rollout_number"] = provenance["rollout_number"]
        # A streaming batch's total is not available at this seam. Do not invent it.
        with ExitStack() as stack:
            client, data = stack.enter_context(
                worker(
                    self.game["godot_binary"],
                    output / "environment",
                    90,
                    (
                        f"--agent-max-episode-ticks={round(self.game['episode_seconds'] * 120) + WARMUP_TICKS}",
                    ),
                    rendered=True,
                )
            )
            trace = stack.enter_context((output / "harness.jsonl").open("w"))
            plans = stack.enter_context(
                (output / "trajectory_decisions.jsonl").open("w")
            )
            decisions = stack.enter_context((output / "decisions.jsonl").open("w"))
            observations = stack.enter_context(
                (output / "observations.jsonl").open("w")
            )
            env = ThunderhillEnv(
                client,
                int(hashlib.sha256(session.encode()).hexdigest()[:8], 16),
                trace,
                lambda: session,
                road_telemetry=self.road,
            )
            env.reset(policy_display=display)
            view = json.loads(env.observe())
            episode_id = env._observation["episode_id"]
            channel = stack.enter_context(
                grpc.insecure_channel(
                    f"{address.ip}:{address.port}",
                    options=[
                        ("grpc.max_send_message_length", 64 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ],
                )
            )
            stub = driver_grpc.EgodriverServiceStub(channel)
            started = False
            failure = None
            metrics = None
            captures = []
            tracker = TrajectoryTracker()
            stall_monitor = StallMonitor()
            baseline = None
            model_decisions = 0
            reason = "episode_tick_limit"

            def capture_and_submit():
                obs = env._observation
                receipt = client.request(
                    dict(op="capture", episode_id=episode_id, expected_tick=obs["tick"])
                )
                if (
                    receipt.get("tick") != obs["tick"]
                    or receipt.get("episode_id") != episode_id
                    or "error" in receipt
                ):
                    raise RuntimeError(
                        f"Invalid image receipt: {receipt.get('error', 'identity mismatch')}"
                    )
                pixels = base64.b64decode(receipt["image"]["base64"], validate=True)
                if hashlib.sha256(pixels).hexdigest() != receipt["image"]["sha256"]:
                    raise ValueError("Image digest mismatch")
                # AlpaGym's native decoder accepts JPEG (including nvJPEG on CUDA).
                # Preserve source PNG plus the exact bytes actually sent to the model.
                encoded = io.BytesIO()
                with Image.open(io.BytesIO(pixels)) as image:
                    image.convert("RGB").save(
                        encoded, format="JPEG", quality=95, subsampling=0
                    )
                jpeg = encoded.getvalue()
                jpeg_digest = hashlib.sha256(jpeg).hexdigest()
                image_directory = output / "driver_images"
                image_directory.mkdir(exist_ok=True)
                (image_directory / f"{jpeg_digest}.jpg").write_bytes(jpeg)
                position, rotation = measured_rig(obs, receipt)
                timestamp = obs["tick"] * 1_000_000 // 120
                captures.append((timestamp, position, rotation, receipt, jpeg))
                observations.write(
                    json.dumps(
                        dict(
                            tick=obs["tick"],
                            timestamp_us=timestamp,
                            observation_id=receipt["observation_id"],
                            image_sha256=receipt["image"]["sha256"],
                            driver_jpeg_sha256=jpeg_digest,
                            jpeg_quality=95,
                            jpeg_subsampling=0,
                            local_position=position.tolist(),
                            local_rotation=rotation.tolist(),
                            camera=receipt["camera"],
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                observations.flush()
                if started:
                    submit(captures[-1])
                return position, rotation

            def submit(row):
                timestamp, position, rotation, receipt, pixels = row
                stub.submit_image_observation(
                    driver.RolloutCameraImage(
                        session_uuid=session,
                        camera_image=driver.RolloutCameraImage.CameraImage(
                            frame_start_us=timestamp,
                            frame_end_us=timestamp,
                            image_bytes=pixels,
                            logical_id=CAMERA_ID,
                        ),
                    ),
                    timeout=120,
                )
                stub.submit_egomotion_observation(
                    driver.RolloutEgoTrajectory(
                        session_uuid=session,
                        trajectory=common.Trajectory(
                            poses=[
                                common.PoseAtTime(
                                    timestamp_us=timestamp,
                                    pose=pose_proto(position, rotation),
                                )
                            ]
                        ),
                    ),
                    timeout=120,
                )

            def advance(controls, source, diagnostics=None):
                nonlocal view
                before = env._observation["tick"]
                command = {
                    key: controls[key]
                    for key in ("throttle", "steer", "front_brake", "rear_brake")
                }
                view = json.loads(
                    env.control_bike(view["observation_token"], **command)
                )
                decisions.write(
                    json.dumps(
                        dict(
                            before_tick=before,
                            tick=env._observation["tick"],
                            action_source=source,
                            controls=command,
                            controller_diagnostics=diagnostics,
                            session_uuid=session,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                decisions.flush()

            try:
                capture_and_submit()
                for _ in range(WARMUP_TICKS // 12):
                    advance(
                        dict(throttle=0.0, steer=0.0, front_brake=1.0, rear_brake=1.0),
                        "stationary_sensor_warmup",
                    )
                    if view["done"]:
                        raise RuntimeError(
                            "Simulator terminated during stationary sensor warmup"
                        )
                    capture_and_submit()
                if env._observation["state"]["speed"] > 0.01:
                    raise RuntimeError("Standing warmup moved the motorcycle")
                baseline = env._observation["track"]["legal_distance"]
                stall_monitor.observe(0, baseline)
                first = captures[0]
                stub.start_session(
                    driver.DriveSessionRequest(
                        session_uuid=session,
                        random_seed=int(
                            hashlib.sha256(session.encode()).hexdigest()[:8], 16
                        ),
                        debug_info=driver.DriveSessionRequest.DebugInfo(
                            scene_id=SCENE_ID
                        ),
                        rollout_spec=driver.DriveSessionRequest.RolloutSpec(
                            vehicle=driver.DriveSessionRequest.RolloutSpec.VehicleDefinition(
                                available_cameras=[
                                    camera_calibration(first[3], first[1], first[2])
                                ]
                            )
                        ),
                    ),
                    timeout=120,
                )
                started = True
                for captured in captures:
                    submit(captured)
                captures[:] = captures[-1:]
                while not view["done"]:
                    if self.shutdown.is_set():
                        raise RuntimeError("Runtime shutdown interrupted this attempt")
                    now, position, rotation, _, _ = captures[-1]
                    obs = env._observation
                    station = obs["track"]["progress"] * self.road.length
                    route_world = np.array(
                        [
                            GODOT_TO_LOCAL @ self.road.point(station + distance)
                            for distance in np.linspace(0.0, 80.0, 20)
                        ]
                    )
                    route_rig = (route_world - position) @ rotation
                    stub.submit_route(
                        driver.RouteRequest(
                            session_uuid=session,
                            route=driver.Route(
                                timestamp_us=now,
                                waypoints=[
                                    common.Vec3(x=p[0], y=p[1], z=p[2])
                                    for p in route_rig
                                ],
                            ),
                        ),
                        timeout=120,
                    )
                    call = stub.drive.future(
                        driver.DriveRequest(
                            session_uuid=session, time_now_us=now, time_query_us=now
                        ),
                        timeout=600,
                    )
                    while True:
                        try:
                            response = call.result(timeout=1)
                            break
                        except grpc.FutureTimeoutError:
                            if self.shutdown.is_set():
                                call.cancel()
                                raise RuntimeError(
                                    "Runtime shutdown interrupted policy inference"
                                )
                    if response.terminate_session:
                        if model_decisions == 0:
                            raise RuntimeError(
                                "Driver terminated without one model decision"
                            )
                        reason = "driver_terminated"
                        break
                    xyz = future_in_rig(response, now, position, rotation)
                    tracker.replan(xyz)
                    model_decisions += 1
                    plans.write(
                        json.dumps(
                            dict(
                                timestamp_us=now,
                                session_uuid=session,
                                xyz_rig=xyz,
                                route_rig=route_rig.tolist(),
                                policy_version=provenance["policy_version"],
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    plans.flush()
                    for _ in range(CONTROL_SAMPLES):
                        state = env._observation["state"]
                        controls, diagnostic = tracker.next_controls(
                            state["speed"], state["lean"]
                        )
                        advance(controls, "model_trajectory_controller", diagnostic)
                        capture_and_submit()
                        obs = env._observation
                        if stall_monitor.observe(
                            obs["tick"] - WARMUP_TICKS, obs["track"]["legal_distance"]
                        ):
                            reason = "stalled"
                            break
                        if view["done"] or not env._observation["track"]["lap_valid"]:
                            break
                    captures[:] = captures[-1:]
                    obs = env._observation
                    if reason == "stalled":
                        break
                    if not obs["track"]["lap_valid"]:
                        reason = "track_limits"
                        break
                    if view["done"]:
                        reason = (
                            obs.get("termination_reason")
                            or obs.get("truncation_reason")
                            or "finished"
                        )
                final = env._observation
                metrics = dict(
                    progress=float(final["track"]["legal_distance"] - baseline),
                    collision_any=float(final["state"]["crashed"]),
                    offroad=float(not final["track"]["lap_valid"]),
                )
                _write_json(
                    output / "summary.json",
                    dict(
                        **provenance,
                        metrics=metrics,
                        stop_reason=reason,
                        final_observation=final,
                        warmup_progress_excluded=True,
                        stall_diagnostic=stall_monitor.diagnostic,
                        model_decisions=model_decisions,
                    ),
                )
                return metrics
            except BaseException as error:
                failure = error
                reason = "infrastructure_failure"
                raise
            finally:
                # Close first so a failed session close cannot be labeled eligible.
                close_failure = None
                if started:
                    try:
                        stub.close_session(
                            driver.DriveSessionCloseRequest(session_uuid=session),
                            timeout=5,
                        )
                    except Exception as close_error:
                        if failure is None:
                            close_failure = failure = close_error
                            reason = "infrastructure_failure"
                        else:
                            failure.add_note(f"Driver close failed: {close_error!r}")
                # Flush and publish recordings even if driver RPC or inference failed.
                try:
                    client.request(
                        dict(op="reset", policy_id="alpagym-recording-closed")
                    )
                    paths = list(data.rglob(f"{episode_id}.jsonl"))
                    if len(paths) != 1:
                        raise RuntimeError(
                            "Expected exactly one closed simulator recording"
                        )
                    enqueue_video(
                        output,
                        paths[0],
                        metadata=dict(
                            **provenance,
                            kind="alpagym_rollout",
                            policy_display=display,
                            stop_reason=reason,
                            decisions="decisions.jsonl",
                            training_eligible=failure is None
                            and not provenance["is_validation"]
                            and not provenance.get("fixture", False),
                        ),
                    )
                except BaseException as archive_error:
                    if failure is None:
                        raise
                    failure.add_note(f"Recording publication failed: {archive_error!r}")
                if close_failure is not None:
                    raise close_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--game-config", type=Path, required=True)
    args = parser.parse_args()
    from alpagym_host.config import load_run_config
    from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint

    cfg = load_run_config(args.resolved_config)
    game = json.loads(args.game_config.read_text())
    service = GodotRuntime(game, Path(cfg.artifact_paths.run_dir) / "rollout_identity")
    server = grpc.server(ThreadPoolExecutor(max_workers=game["concurrency"] + 4))
    runtime_grpc.add_RuntimeServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        raise RuntimeError("Unable to bind runtime service")
    server.start()
    signal.signal(signal.SIGTERM, lambda *_: service.shutdown.set())
    signal.signal(signal.SIGINT, lambda *_: service.shutdown.set())
    try:
        FileTopologyRegistry(
            cfg.artifact_paths.topology_registry_dir
        ).publish_alpasim_runtime(
            TopologyEndpoint(
                id="thunderhill-godot",
                host="127.0.0.1",
                port=port,
                capacity=game["concurrency"],
            )
        )
        while not service.shutdown.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(grace=30).wait()


if __name__ == "__main__":
    main()
