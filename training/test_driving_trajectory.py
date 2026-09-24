import math
import os
import json
import tempfile
from pathlib import Path
import unittest

from driving_trajectory import (
    TrajectoryTracker, encode_controller_controls,
    decode_controller_controls,
)


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
                state = episode.observation['state']
                x, y, z = state['position']
                position = [-z, -x, y]
                heading = state['heading']
                forward = [math.cos(heading), -math.sin(heading), 0]
                if index % 5 == 0:
                    tracker.replan([[p + f * target * .1 * i for p, f in zip(position, forward)]
                                    for i in range(1, 6)], origin=position)
                controls, diagnostics = tracker.next_controls(state['speed'], position=position, forward=forward)
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
