"""Real camera and controller plumbing, explicitly not a learned policy run."""
import os
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from driving_episode import DrivingEpisode


@unittest.skipUnless(os.environ.get('THUNDERHILL_GODOT'), 'Rendered Godot required')
class DrivingEpisodeTests(unittest.TestCase):
    def test_moving_start_uses_real_history_and_excludes_setup_reward(self):
        root = Path(tempfile.mkdtemp(prefix='moving-driving-integration-',
                                    dir=Path(__file__).resolve().parents[1] / 'artifacts'))
        with DrivingEpisode(godot=os.environ['THUNDERHILL_GODOT'], output=root / 'episode',
                adapter_sha256='0' * 64, model='synthetic moving start plumbing test',
                revision='not a model', generation=0, rollout=1, evaluation=True,
                initial_speed_m_s=5.0, time_budget_seconds=0.5) as run:
            observation = run.model_input()
            self.assertEqual(observation['tick'], 600)
            self.assertEqual(observation['images'].shape, (4, 360, 640, 3))
            self.assertEqual(len(set(run.image_ids)), 4)
            self.assertGreater(np.linalg.norm(observation['ego_history_xyz'][0]), 1)
            self.assertEqual(len(run.episode.records), 50)
            self.assertTrue(all(row['action_source'] == 'scenario_setup' for row in run.episode.records))
            self.assertLess(run.setup_speed_metrics['max_m_s'] - run.setup_speed_metrics['min_m_s'], .15)
            self.assertAlmostEqual(run.setup_speed_metrics['final_m_s'], 5., delta=.15)
            start_progress = run.episode.observation['track']['legal_distance']
            self.assertGreater(start_progress, 1)
            plan = np.zeros((64, 3))
            plan[:, 0] = np.arange(1, 65) * .5
            run.execute(dict(xyz=plan, replay={'fixture': True}, old_logprob=0))
            summary = run.finish()
            summary['training_eligible'] = False
            (run.episode.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            self.assertEqual(summary['recorded_transitions'], 660)
            self.assertEqual(summary['model_control_start_tick'], 600)
            self.assertAlmostEqual(summary['reward_components']['sim_seconds'], .5)
            self.assertAlmostEqual(summary['reward_components']['legal_progress_m'],
                                   summary['final_observation']['track']['legal_distance'] - start_progress)
            self.assertEqual(summary['trajectory_decisions'], 1)
            self.assertTrue(summary['recording_provenance_verified'])
        print('Rendered moving start artifacts:', root)

    def test_camera_controller_reward_and_recording(self):
        root = Path(tempfile.mkdtemp(prefix='driving-integration-',
                                    dir=Path(__file__).resolve().parents[1] / 'artifacts'))
        with DrivingEpisode(godot=os.environ['THUNDERHILL_GODOT'], output=root / 'episode',
                adapter_sha256='0' * 64, model='synthetic controller plumbing test',
                revision='not a model', generation=0, rollout=1, evaluation=True,
                time_budget_seconds=0.5) as run:
            observation = run.model_input()
            self.assertEqual(observation['images'].shape, (4, 360, 640, 3))
            self.assertEqual(observation['ego_history_xyz'].shape, (16, 3))
            np.testing.assert_array_equal(observation['ego_history_xyz'], 0)
            plan = np.zeros((64, 3))
            plan[:, 0] = np.arange(1, 65) * .5
            run.execute(dict(xyz=plan, replay={'fixture': True}, old_logprob=0))
            summary = run.finish()
            # Synthetic controller tests cannot enter any training campaign.
            summary['training_eligible'] = False
            (run.episode.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            self.assertEqual(summary['recorded_transitions'], 60)
            self.assertTrue(summary['recording_provenance_verified'])
            self.assertGreater(summary['reward_components']['legal_progress_m'], 0)
            self.assertEqual(len(list((run.episode.output / 'video_jobs').glob('*.json'))), 1)
            self.assertTrue((run.episode.output / 'trajectory_decisions.jsonl').is_file())
            replay = run.episode.output / 'trajectory_replays/0000.pt'
            self.assertEqual(torch.load(replay, weights_only=True), {'fixture': True})
        print('Rendered controller test artifacts:', root)


if __name__ == '__main__':
    unittest.main()
