import math
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


if __name__ == "__main__":
    unittest.main()
