import math
import os
import json
import tempfile
from pathlib import Path
import unittest

from driving_trajectory import (
    TrajectoryTracker, godot_history_to_ego, encode_controller_controls,
    decode_controller_controls,
)


class TrajectoryTests(unittest.TestCase):
    def test_continuous_commands_preserve_small_distinct_inputs(self):
        controls = dict(steer=-1.234567891234567e-8, throttle=0.00089031472971384,
                        front_brake=0.000136217883843274, rear_brake=0.0, shift=0)
        self.assertEqual(decode_controller_controls(encode_controller_controls(controls)), controls)
        second = dict(controls, throttle=controls['throttle'] * 1.01)
        self.assertNotEqual(encode_controller_controls(controls), encode_controller_controls(second))

    def test_continuous_commands_reject_invalid_controls(self):
        controls = dict(steer=0.0, throttle=0.0, front_brake=0.0, rear_brake=0.0)
        for value in (float('nan'), float('inf'), True, -0.01, 1.01):
            with self.assertRaises(ValueError):
                encode_controller_controls(dict(controls, throttle=value))
        with self.assertRaises(ValueError):
            decode_controller_controls('control_bike 0 0 0 0')

    def controls(self, plan, speed=10, lean=0):
        tracker = TrajectoryTracker()
        tracker.replan(plan)
        return tracker.next_controls(speed, lean)

    def test_turn_sign_matches_bike(self):
        left, _ = self.controls([(i, i * i * 0.02, 0) for i in range(1, 20)])
        right, _ = self.controls([(i, -i * i * 0.02, 0) for i in range(1, 20)])
        self.assertLess(left["steer"], 0)
        self.assertAlmostEqual(left["steer"], -right["steer"])

    def test_speed_is_from_waypoint_time(self):
        plan = [(i, 0, 0) for i in range(1, 10)]
        slow, diagnostics = self.controls(plan, speed=5)
        fast, _ = self.controls(plan, speed=15)
        exact, _ = self.controls(plan, speed=10)
        self.assertEqual(diagnostics["target_speed_m_s"], 10)
        self.assertGreater(slow["throttle"], 0)
        self.assertEqual(slow["front_brake"], 0)
        self.assertGreater(fast["front_brake"], 0)
        self.assertEqual(fast["throttle"], 0)
        self.assertEqual(exact["steer"], 0)

    def test_stationary_and_reverse_never_force_movement(self):
        for plan in ([(0, 0, 0)] * 4, [(-1, 0, 0)]):
            stopped, _ = self.controls(plan, speed=0)
            moving, _ = self.controls(plan, speed=10)
            self.assertEqual(stopped["throttle"], 0)
            self.assertEqual(moving["throttle"], 0)
            self.assertGreater(moving["front_brake"], 0)

    def test_plan_lifetime_and_replan(self):
        tracker = TrajectoryTracker()
        with self.assertRaises(ValueError):
            tracker.next_controls(0, 0)
        tracker.replan([(1, 0, 0), (2, 0, 0)])
        tracker.next_controls(10, 0)
        controls, diagnostic = tracker.next_controls(10, 0)
        self.assertAlmostEqual(tracker.x, 1)
        self.assertEqual(controls["steer"], 0)
        self.assertEqual(diagnostic["plan_time_s"], 0.1)
        with self.assertRaises(ValueError):
            tracker.next_controls(10, 0)
        tracker.replan([(0, 0, 0)])
        self.assertEqual(tracker.x, 0)
        self.assertEqual(tracker.step, 0)

    def test_invalid(self):
        for plan in ([], [(1, 2)], [(math.nan, 0, 0)], [(0, math.inf, 0)]):
            with self.assertRaises(ValueError):
                self.controls(plan)
        for speed, lean in ((-1, 0), (math.nan, 0), (1, math.inf), (1, math.pi)):
            with self.assertRaises(ValueError):
                self.controls([(1, 0, 0)], speed, lean)

    def test_integral_survives_replan_but_not_stop_or_reverse(self):
        tracker = TrajectoryTracker()
        for _ in range(10):
            tracker.replan([(0.5 * i, 0, 0) for i in range(1, 6)])
            tracker.next_controls(4, 0)
        self.assertGreater(tracker.speed_integral, .1)
        tracker.replan([(0.5, 0, 0)])
        controls, _ = tracker.next_controls(5, 0)
        self.assertGreater(controls['throttle'], .1)
        for plan in ([(0, 0, 0)], [(-1, 0, 0)]):
            tracker.replan(plan)
            controls, _ = tracker.next_controls(0, 0)
            self.assertEqual(controls['throttle'], 0)
            self.assertEqual(tracker.speed_integral, 0)

    def test_integral_does_not_wind_up_at_saturation(self):
        tracker = TrajectoryTracker()
        for _ in range(100):
            tracker.replan([(100, 0, 0)])
            controls, _ = tracker.next_controls(0, 0)
            self.assertEqual(controls['throttle'], 1)
        self.assertEqual(tracker.speed_integral, 0)

    def test_history_forward_left_up(self):
        xyz, rotations = godot_history_to_ego(
            [(0, 0, 1), (-2, 3, 0), (0, 0, 0)], [0, 0, 0])
        self.assertEqual(xyz[0], [-1, 0, 0])
        self.assertEqual(xyz[1], [0, 2, 3])
        self.assertEqual(xyz[-1], [0, 0, 0])
        self.assertEqual(rotations[-1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_history_rotated_rig(self):
        xyz, rotations = godot_history_to_ego([(0, 0, 0), (1, 0, 0)], [0, math.pi / 2])
        self.assertAlmostEqual(xyz[0][0], -1)
        self.assertAlmostEqual(xyz[0][1], 0)
        # Old forward (-Godot Z) is left relative to the new +Godot X forward.
        self.assertAlmostEqual(rotations[0][1][0], 1)
        self.assertAlmostEqual(rotations[0][0][1], -1)

    def test_history_invalid(self):
        with self.assertRaises(ValueError):
            godot_history_to_ego([(0, 0, 0)], [])


@unittest.skipUnless(os.environ.get('THUNDERHILL_GODOT'), 'Native Godot required')
class NativeSpeedControllerTests(unittest.TestCase):
    def exercise(self, name, initial_speed, targets):
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
                if index % 5 == 0:
                    tracker.replan([(target * .1 * i, 0, 0) for i in range(1, 6)])
                state = episode.observation['state']
                controls, diagnostics = tracker.next_controls(state['speed'], state['lean'])
                episode.apply(encode_controller_controls(controls), [], [])
                rows.append(dict(tick=episode.observation['tick'], target=target,
                                 speed=episode.observation['state']['speed'],
                                 controls=controls, diagnostics=diagnostics))
                if episode.done:
                    break
            summary = episode.finish()
            summary['training_eligible'] = False
            (episode.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (root / 'controller.json').write_text(json.dumps(rows, indent=2) + '\n')
        self.assertTrue(summary['recording_provenance_verified'])
        self.assertEqual(len(rows), len(targets))
        print('Native speed controller artifacts:', root)
        return rows

    def test_constant_speed_holds_through_replans(self):
        rows = self.exercise('constant', 5, [5] * 100)
        self.assertLess(max(abs(row['speed'] - 5) for row in rows[-50:]), .15)
        self.assertGreater(min(row['speed'] for row in rows), 4)
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
