import math
import unittest

from driving_trajectory import TrajectoryTracker, godot_history_to_ego


class TrajectoryTests(unittest.TestCase):
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
