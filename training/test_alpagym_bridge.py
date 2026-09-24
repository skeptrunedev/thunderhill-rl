"""GPU free protocol tests. Scripted driver is a test fixture, never training data."""

import json
import math
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest

import grpc
import numpy as np
from scipy.spatial.transform import Rotation
from training.alpagym_bridge import (
    ROOT,
    CAMERA_ID,
    SCENE_ID,
    GODOT_TO_LOCAL,
    GodotRuntime,
    measured_rig,
    future_in_rig,
    pose_proto,
    common,
    driver,
    driver_grpc,
    runtime,
    runtime_grpc,
)


class GeometryTests(unittest.TestCase):
    def test_camera_roll_and_pitch_are_preserved(self):
        rig = Rotation.from_euler("zyx", [0.7, 0.12, -0.3]).as_matrix()
        tangent, left, up = (GODOT_TO_LOCAL.T @ rig).T
        forward = tangent * math.cos(0.08) - up * math.sin(0.08)
        camera_up = tangent * math.sin(0.08) + up * math.cos(0.08)
        receipt = {
            "camera": {
                "pose": {
                    "basis_x": (-left).tolist(),
                    "basis_y": camera_up.tolist(),
                    "basis_z": (-forward).tolist(),
                }
            }
        }
        position, actual = measured_rig(
            {"state": {"position": [3.0, 4.0, 5.0]}}, receipt
        )
        np.testing.assert_allclose(actual, rig, atol=1e-10)
        np.testing.assert_allclose(position, [-5.0, -3.0, 4.0])

    def test_native_world_plan_current_pose_removed(self):
        rotation = Rotation.from_euler("z", math.pi / 2).as_matrix()
        origin = np.array([50.0, 70.0, 3.0])
        response = driver.DriveResponse(
            trajectory=common.Trajectory(
                poses=[
                    common.PoseAtTime(
                        timestamp_us=1_500_000 + i * 100_000,
                        pose=pose_proto(
                            origin + rotation @ np.array([i * 0.5, 0.0, 0.0]), rotation
                        ),
                    )
                    for i in range(4)
                ]
            )
        )
        actual = future_in_rig(response, 1_500_000, origin, rotation)
        np.testing.assert_allclose(
            actual, [[0.5, 0, 0], [1.0, 0, 0], [1.5, 0, 0]], atol=1e-6
        )
        response.trajectory.poses[-1].timestamp_us += 1
        with self.assertRaises(ValueError):
            future_in_rig(response, 1_500_000, origin, rotation)


class FixtureDriver(driver_grpc.EgodriverServiceServicer):
    """Constant speed geometric fixture, not a policy, demonstration or RL run."""

    def __init__(self, fail=False, speed=5.0):
        self.fail = fail
        self.speed = speed
        self.images = []
        self.poses = []
        self.routes = []
        self.drive_times = []
        self.closed = False
        self.started = None
        self.ground_truth_calls = 0

    def start_session(self, request, context):
        self.started = request
        return common.SessionRequestStatus()

    def submit_image_observation(self, request, context):
        self.images.append(request)
        return common.Empty()

    def submit_egomotion_observation(self, request, context):
        self.poses.extend(request.trajectory.poses)
        return common.Empty()

    def submit_route(self, request, context):
        self.routes.append(request)
        return common.Empty()

    def submit_recording_ground_truth(self, request, context):
        self.ground_truth_calls += 1
        raise AssertionError("Ground truth must never be submitted")

    def drive(self, request, context):
        self.drive_times.append(request.time_now_us)
        if self.fail:
            context.abort(grpc.StatusCode.INTERNAL, "injected fixture failure")
        pose = self.poses[-1].pose
        origin = np.array([pose.vec.x, pose.vec.y, pose.vec.z])
        q = pose.quat
        rotation = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        return driver.DriveResponse(
            trajectory=common.Trajectory(
                poses=[
                    common.PoseAtTime(
                        timestamp_us=request.time_now_us + i * 100_000,
                        pose=pose_proto(
                            origin
                            + rotation @ np.array([i * 0.1 * self.speed, 0.0, 0.0]),
                            rotation,
                        ),
                    )
                    for i in range(65)
                ]
            )
        )

    def close_session(self, request, context):
        self.closed = True
        return common.Empty()


@unittest.skipUnless(
    os.environ.get("THUNDERHILL_GODOT"),
    "Set THUNDERHILL_GODOT for rendered Godot contract test",
)
class RealGameTests(unittest.TestCase):
    def run_fixture(self, fail, fixture=None, *, seconds=0.6, validation=True):
        output = Path(tempfile.mkdtemp(prefix="thunderhill-alpagym-contract-"))
        fixture = fixture or FixtureDriver(fail)
        server = grpc.server(ThreadPoolExecutor(max_workers=4))
        driver_grpc.add_EgodriverServiceServicer_to_server(fixture, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        service = GodotRuntime(
            dict(
                godot_binary=os.environ["THUNDERHILL_GODOT"],
                project_path=str(ROOT / "godot"),
                episode_seconds=seconds,
                concurrency=1,
                recording_root=str(output),
            ),
            output / "unused",
            identity_provider=lambda port: dict(
                policy_version=7,
                batch_id="fixture-not-training",
                is_validation=validation,
                active=True,
                fixture=True,
            ),
        )
        runtime_server = grpc.server(ThreadPoolExecutor(max_workers=4))
        runtime_grpc.add_RuntimeServiceServicer_to_server(service, runtime_server)
        runtime_port = runtime_server.add_insecure_port("127.0.0.1:0")
        runtime_server.start()
        try:
            with grpc.insecure_channel(f"127.0.0.1:{runtime_port}") as channel:
                stub = runtime_grpc.RuntimeServiceStub(channel)
                info = stub.get_runtime_info(common.Empty())
                self.assertEqual(info.scenes[0].scene_id, SCENE_ID)
                result = stub.simulate(
                    runtime.SimulationRequest(
                        available_drivers=[
                            runtime.SimulationRequest.DriverAddress(
                                ip="127.0.0.1", port=port
                            )
                        ],
                        rollout_specs=[
                            runtime.RolloutSpec(
                                scenario_id=SCENE_ID,
                                nr_rollouts=1,
                                session_uuids=["fixture"],
                            )
                        ],
                        n_concurrent_per_driver=1,
                    ),
                    timeout=180,
                ).rollout_returns[0]
            return output / "fixture", fixture, result
        finally:
            runtime_server.stop(0).wait()
            server.stop(0).wait()

    def test_real_game_sensor_plan_control_and_recording(self):
        output, fixture, result = self.run_fixture(False)
        self.assertTrue(result.success, result.error)
        self.assertEqual(
            set(result.aggregated_metrics),
            {"progress", "collision_any", "offroad", "fall_without_collision"},
        )
        self.assertGreater(result.aggregated_metrics["progress"], 0)
        self.assertEqual(fixture.drive_times, [1_500_000, 1_700_000, 1_900_000])
        self.assertEqual(
            [p.timestamp_us for p in fixture.poses[:16]],
            list(range(0, 1_500_001, 100_000)),
        )
        self.assertEqual(fixture.images[0].camera_image.logical_id, CAMERA_ID)
        self.assertEqual(len(fixture.routes[0].route.waypoints), 20)
        self.assertEqual(fixture.ground_truth_calls, 0)
        self.assertTrue(fixture.closed)
        self.assertEqual(len(list((output / "video_jobs").glob("*.json"))), 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["policy_version"], 7)
        self.assertTrue(summary["warmup_progress_excluded"])
        definition = summary["reward_definition"]
        self.assertAlmostEqual(
            result.aggregated_metrics["progress"],
            definition["legal_progress_m"] / definition["progress_normalizer_m"],
        )
        self.assertGreater(definition["progress_normalizer_m"], 1000)
        self.assertEqual(result.aggregated_metrics["offroad"], 0)
        self.assertEqual(result.aggregated_metrics["fall_without_collision"], 0)

    def test_official_driver_and_policy_preprocessing(self):
        from concurrent.futures import Future
        import torch
        from alpagym_host.config import (
            AlpamayoPolicyConfig,
            ModelConfig,
            InferenceConfig,
            SamplingParamsConfig,
            TrajectorySelectorKind,
        )
        from alpagym_runtime.alpasim.driver_server import EgodriverGrpcServicer
        from alpagym_runtime.policies.alpamayo.policy import AlpamayoPolicy
        from alpagym_runtime.inference.types import ModelOutput

        class FixtureInference:
            def __init__(self):
                self.inputs = []

            def infer(self, inputs):
                self.inputs.append(inputs)
                result = Future()
                xyz = torch.zeros((1, 1, 64, 3))
                xyz[0, 0, :, 0] = torch.arange(1, 65) * 0.5
                result.set_result(
                    ModelOutput(
                        pred_xyz=xyz,
                        pred_rot=torch.eye(3).expand(1, 1, 64, 3, 3).clone(),
                    )
                )
                return result

        engine = FixtureInference()
        config = AlpamayoPolicyConfig(
            kind="alpamayo",
            model=ModelConfig(
                kind="alpamayo_r1",
                path="fixture-no-weights",
                device="cpu",
                dtype="float32",
                use_cameras=[CAMERA_ID],
                num_context_frames=4,
                num_historical_waypoints=16,
                num_future_waypoints=64,
                step_dt_us=100_000,
                input_size=[320, 512],
            ),
            inference=InferenceConfig(
                max_batch_size=1,
                return_trace_for_rl=False,
                sampling=SamplingParamsConfig(
                    top_p=0.98,
                    top_k=None,
                    temperature=0.6,
                    num_traj_samples=1,
                    num_traj_sets=1,
                ),
            ),
            trajectory_selector=TrajectorySelectorKind.identity,
        )
        fixture = EgodriverGrpcServicer(
            lambda session, calibration, seed: AlpamayoPolicy(
                engine, session, config, torch.device("cpu"), torch.float32, seed
            )
        )
        output, _, result = self.run_fixture(False, fixture)
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(engine.inputs), 3)
        first = engine.inputs[0]
        self.assertEqual(tuple(first.camera_frames.shape), (4, 3, 320, 512))
        self.assertEqual(
            first.relative_timestamps.tolist(), [-300_000, -200_000, -100_000, 0]
        )
        self.assertEqual(tuple(first.ego_history_xyz.shape), (1, 16, 3))
        self.assertEqual(tuple(first.route_xy.shape), (20, 2))
        self.assertTrue(torch.isfinite(first.route_xy).all())
        self.assertTrue(
            torch.allclose(first.ego_history_xyz[0, -1], torch.zeros(3), atol=1e-6)
        )
        record = fixture.pop_session_record("fixture")
        self.assertEqual(len(record.outputs), 3)
        self.assertIsNone(record.ground_truth)
        print(f"Official driver Godot recording: {output}")

    def test_rollout_identity_without_invented_count(self):
        output, _, result = self.run_fixture(False, validation=False)
        self.assertTrue(result.success, result.error)
        job = json.loads(next((output / "video_jobs").glob("*.json")).read_text())
        self.assertEqual(job["policy_display"]["generation"], 7)
        self.assertEqual(job["policy_display"]["rollout_number"], 1)
        self.assertNotIn("rollout_count", job["policy_display"])
        self.assertFalse(job["metadata"]["training_eligible"])
        print(f"Rollout overlay fixture recording: {output}")

    def test_stalled_attempt_stops_with_valid_metrics(self):
        output, _, result = self.run_fixture(
            False, FixtureDriver(speed=0.0), seconds=12.0
        )
        self.assertTrue(result.success, result.error)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["stop_reason"], "stalled")
        self.assertEqual(summary["final_observation"]["tick"], 1380)
        self.assertLess(summary["metrics"]["progress"], 1.0)
        self.assertTrue(summary["stall_diagnostic"]["stalled"])

    def test_driver_failure_still_preserves_video(self):
        output, fixture, result = self.run_fixture(True)
        self.assertFalse(result.success)
        self.assertIn("injected fixture failure", result.error)
        self.assertTrue(fixture.closed)
        jobs = list((output / "video_jobs").glob("*.json"))
        self.assertEqual(len(jobs), 1)
        self.assertFalse(
            json.loads(jobs[0].read_text())["metadata"]["training_eligible"]
        )


if __name__ == "__main__":
    unittest.main()
