"""Record parsing retains failures and rejects unrecognized rows."""
import io
import json
import unittest

from lap_rollout import recorded_transitions, recording_rows


class RecordingTests(unittest.TestCase):
    def source(self, rows):
        return io.StringIO(''.join(json.dumps(row) + '\n' for row in rows))

    def test_metadata_is_not_a_physics_transition(self):
        rows = [{'type': 'camera_observation'}, {'type': 'model_decision'},
                {'type': 'snapshot'}, {'type': 'transition', 'tick': 1},
                {'type': 'transition', 'tick': 2, 'state': {'crashed': True}}]
        self.assertEqual(list(recording_rows(self.source(rows))), rows)
        self.assertEqual(list(recorded_transitions(self.source(rows))), rows[3:])

    def test_unknown_or_malformed_records_raise(self):
        for row in ({'type': 'unknown'}, {}, [], 1, None):
            with self.subTest(row=row), self.assertRaises(ValueError):
                list(recorded_transitions(self.source([row])))
        with self.assertRaises(json.JSONDecodeError):
            list(recording_rows(io.StringIO('not json\n')))


if __name__ == '__main__':
    unittest.main()
