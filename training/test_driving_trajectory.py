import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from driving_trajectory import (
    TrajectoryTracker,
    decode_controller_controls,
    encode_controller_controls,
)


@unittest.skipUnless(os.environ.get('THUNDERHILL_GODOT'), 'Native Godot required')
class NativeSpeedControllerTests(unittest.TestCase):
    def exercise(self, name, initial_speed, targets, curvature=0.0, recorded_plan=None, plan_source=None):
        from lap_episode import LapEpisode
        from lap_policy import RoadTelemetry
        root = Path(tempfile.mkdtemp(prefix='speed-controller-' + name + '-',
                                    dir=Path(__file__).resolve().parents[1] / 'artifacts'))
        tracker = TrajectoryTracker()
        rows = []
        with LapEpisode(godot=os.environ['THUNDERHILL_GODOT'], output=root / 'episode',
                road=RoadTelemetry(), adapter_sha256='0' * 64,
                model='synthetic controller verification', revision='not a model',
                generation=0, rollout=1, evaluation=True, initial_speed_m_s=initial_speed,
                action_parser=decode_controller_controls,
                time_budget_seconds=len(targets) * .1) as episode:
            for index, target in enumerate(targets):
                # Reset the trajectory frame repeatedly, as inference does.
                state = episode.observation['state']
                x, y, z = state['position']
                position = [-z, -x, y]
                heading = state['heading']
                forward = [math.cos(heading), -math.sin(heading), 0]
                if index == 0 or (recorded_plan is None and index % 2 == 0):
                    points = []
                    for i in range(1, 65):
                        distance = target * .1 * i
                        x = math.sin(curvature * distance) / curvature if curvature else distance
                        y = (1 - math.cos(curvature * distance)) / curvature if curvature else 0
                        points.append([position[0] + forward[0] * x - forward[1] * y,
                                       position[1] + forward[1] * x + forward[0] * y, position[2]])
                    if recorded_plan is not None:
                        points = [[position[0] + forward[0] * x - forward[1] * y,
                                   position[1] + forward[1] * x + forward[0] * y,
                                   position[2] + z] for x, y, z in recorded_plan]
                    tracker.replan(points, origin=position)
                controls, diagnostics = tracker.next_controls(state['speed'], position=position, forward=forward)
                episode.apply(encode_controller_controls(controls), [], [])
                rows.append({'tick': episode.observation['tick'], 'target': target,
                                 'speed': episode.observation['state']['speed'],
                                 'lean': episode.observation['state']['lean'],
                                 'heading': episode.observation['state']['heading'],
                                 'position': episode.observation['state']['position'],
                                 'controls': controls, 'diagnostics': diagnostics})
                if episode.done:
                    break
            summary = episode.finish()
            summary['training_eligible'] = False
            (episode.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (root / 'controller.json').write_text(json.dumps(rows, indent=2) + '\n')
        (root / 'verification.json').write_text(json.dumps({
            'training_eligible': False,
            'source_kind': 'archived_model_output' if plan_source else 'synthetic_actuator_fixture',
            'source_path': str(plan_source) if plan_source else None,
            'source_sha256': hashlib.sha256(plan_source.read_bytes()).hexdigest() if plan_source else None,
            'max_speed_error_m_s': max(abs(row['speed'] - row['target']) for row in rows),
            'max_position_residual_m': max(row['diagnostics']['tracking_position_error_m'] for row in rows),
            'measured_elevation_range_m': max(row['position'][1] for row in rows) - min(row['position'][1] for row in rows),
        }, indent=2) + '\n')
        self.assertTrue(summary['recording_provenance_verified'])
        self.assertEqual(len(rows), len(targets))
        print('Native speed controller artifacts:', root)
        return rows

    def test_measured_world_pose_tracks_mirrored_curves(self):
        for curvature in (-0.02, 0.02):
            rows = self.exercise('curve', 10, [10] * 12, curvature=curvature)
            self.assertTrue(all(row['diagnostics']['feedback'] == 'measured_world_pose' for row in rows))
            self.assertGreater(abs(rows[-1]['lean']), .03)
            self.assertLess(rows[-1]['lean'] * curvature, 0)
            self.assertLess(abs(rows[-1]['speed'] - 10), .5)

    def test_low_speed_uses_wheel_angle_for_mirrored_curves(self):
        for curvature in (-0.02, 0.02):
            rows = self.exercise('slow-curve', 2, [2] * 30, curvature=curvature)
            self.assertTrue(all(row['diagnostics']['steering_mode'] == 'wheel_angle' for row in rows))
            delta = rows[-1]['heading'] - rows[0]['heading']
            heading_change = math.atan2(math.sin(delta), math.cos(delta))
            self.assertLess(heading_change * curvature, 0)
            self.assertGreater(abs(heading_change), .05)
            self.assertLess(abs(rows[-1]['speed'] - 2), .2)

    @unittest.skipUnless(os.environ.get('THUNDERHILL_RECORDED_MODEL_PLAN'),
                         'Archived model trajectory must be explicitly supplied')
    def test_archived_model_plan_actuation(self):
        # This is inference output replay for actuator qualification, never SFT
        # or an RL sample. Keep the original source path beside the recording.
        source = Path(os.environ['THUNDERHILL_RECORDED_MODEL_PLAN']).resolve()
        plan = json.loads(source.read_text().splitlines()[0])['xyz']
        segments = list(zip([[0, 0, 0], *plan[:-1]], plan))
        targets = [math.dist(a, b) / .1 for a, b in segments]
        rows = self.exercise('archived-model', targets[0], targets, recorded_plan=plan, plan_source=source)
        print('Archived policy trajectory source:', source)
        self.assertLess(max(abs(row['speed'] - row['target']) for row in rows), 1.0)
        self.assertLess(max(row['diagnostics']['tracking_position_error_m'] for row in rows), 1.0)

    def test_constant_speed_holds_through_replans(self):
        rows = self.exercise('constant', 5, [5] * 100)
        self.assertLess(max(abs(row['speed'] - 5) for row in rows[-50:]), .15)
        self.assertGreater(min(row['speed'] for row in rows), 4)
        self.assertGreater(max(row['position'][1] for row in rows) - min(row['position'][1] for row in rows), .1)
        history_speeds = [row['speed'] for row in rows[34:50]]
        self.assertLess(max(history_speeds) - min(history_speeds), .15)
        self.assertLess(abs(history_speeds[-1] - 5), .15)

    def test_acceleration_then_stopping(self):
        rows = self.exercise('accelerate-stop', 0, [5] * 70 + [0] * 50)
        self.assertGreater(rows[69]['speed'], 4.8)
        self.assertLess(rows[-1]['speed'], .1)
        self.assertTrue(all(row['controls']['throttle'] == 0 for row in rows[70:]))


if __name__ == "__main__":
    unittest.main()
