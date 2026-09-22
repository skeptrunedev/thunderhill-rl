import json
import tempfile
import unittest
from pathlib import Path
from render_run_video import inspect_recording, render_video


class RecordingVideoTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'recording.jsonl'
        self.rows = [{'type': 'episode', 'physics_dt': 1/120,
                      'initial_state': {'tick': 24, 'elapsed': 0.2},
                      'policy_display': {'model_name': 'model', 'generation': 4,
                                         'rollout_number': 2, 'rollout_total': 32}}]

    def inspect(self):
        self.path.write_text('\n'.join(json.dumps(row) for row in self.rows))
        return inspect_recording(self.path)

    def test_zero_transition_failure_retained(self):
        self.rows.append({'type': 'environment_failure', 'reason': 'failed'})
        info = self.inspect()
        self.assertEqual(info['expected_frames'], 30)
        self.assertEqual(info['transitions'], 0)
        self.assertEqual(info['policy_display']['generation'], 4)

    def test_partial_last_frame_has_complete_tail(self):
        self.rows.append({'type': 'transition', 'previous_tick': 24, 'tick': 25,
                          'state': {'tick': 25, 'elapsed': 0.2 + 1/120}})
        info = self.inspect()
        self.assertEqual(info['expected_frames'], 31)
        self.assertEqual(info['final_tick'], 25)
        self.rows[-1]['previous_tick'] = 12
        with self.assertRaises(ValueError):
            self.inspect()

    def test_wrong_timestep_and_unknown_row_rejected(self):
        self.rows[0]['physics_dt'] = 1/60
        with self.assertRaises(ValueError):
            self.inspect()
        self.rows[0]['physics_dt'] = 1/120
        self.rows.append({'type': 'unrecognized'})
        with self.assertRaises(ValueError):
            self.inspect()

    def test_no_overwrite(self):
        self.inspect()
        output = self.path.with_suffix('.mp4')
        output.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            render_video(self.path, output, godot='missing', ffmpeg='missing')
        self.assertEqual(output.read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
