"""Staged experiment must gate spending and compare identical evaluation seeds."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from train_alpamayo_rl import baseline_gate, run_diagnostic, collect, SCENARIOS


def summary(scenario, seed, *, progress=50., speed=4.):
    return dict(scenario_id=scenario, sample_seed=seed, recording_provenance_verified=True,
                training_eligible=True, reward_components=dict(total=progress/100, legal_progress_m=progress),
                final_observation=dict(state=dict(speed=speed, crashed=False)), offtrack_ticks=0,
                reason='time_limit', trajectory_metrics=[dict(horizon_x_m=15.)])


class GateTests(unittest.TestCase):
    def baseline(self):
        return [summary(s['scenario_id'], seed) for s in SCENARIOS for seed in (1073, 2073)]

    def test_standing_failure_does_not_mask_moving_results(self):
        rows = self.baseline()
        rows[0].update(reason='stalled')
        self.assertTrue(baseline_gate(rows)['passed'])
        rows[2]['final_observation']['state']['speed'] = .01
        self.assertFalse(baseline_gate(rows)['passed'])

    def test_invalid_motion_recording_and_reverse_plans_fail(self):
        for field, value in [('progress', .1), ('speed', .01)]:
            rows = self.baseline()
            rows[2] = summary('moving', 1073, **{field: value})
            self.assertFalse(baseline_gate(rows)['passed'])
        rows = self.baseline()
        rows[3]['trajectory_metrics'][0]['horizon_x_m'] = -10.
        self.assertFalse(baseline_gate(rows)['passed'])
        rows = self.baseline()
        rows[3]['recording_provenance_verified'] = False
        self.assertFalse(baseline_gate(rows)['passed'])
        with self.assertRaises(ValueError):
            baseline_gate(rows[:3])


class CampaignTests(unittest.TestCase):
    def run_campaign(self, baseline_passes):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        args = SimpleNamespace(output=root, seconds=30, rollouts=8)
        manifest = dict(complete=False, generations=[], evaluations=[])
        calls, updates, logs = [], [], []

        class Policy:
            current = 'initial'
            saved = {root / 'initial_adapter': 'initial'}

            def update(self, episodes, max_decisions):
                updates.append(copy.deepcopy(episodes))
                self.current = 'updated' + str(len(updates))
                return dict(optimizer_steps=1, parameter_delta_l1=1.)

            def save(self, path):
                self.saved[path] = self.current
                return self.current

            def reload(self, path):
                self.current = self.saved[path]

            def fingerprint(self):
                return self.current

        def collect(policy, args, directory, **kwargs):
            directory.mkdir()
            calls.append(kwargs)
            row = summary(kwargs['scenario_id'], kwargs['seed'],
                          speed=4. if baseline_passes else .01)
            return dict(reward=.5, reward_group=kwargs['scenario_id'], replays=['test']), row

        tracker = SimpleNamespace(log=logs.append)
        with patch('train_alpamayo_rl.collect', collect):
            run_diagnostic(Policy(), args, tracker, manifest, 'initial')
        return manifest, calls, updates

    def test_failed_baseline_never_updates_or_collects_training(self):
        manifest, calls, updates = self.run_campaign(False)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(c['evaluation'] for c in calls))
        self.assertEqual(updates, [])
        self.assertTrue(manifest['diagnostic_complete'])
        self.assertFalse(manifest['complete'])
        self.assertEqual(manifest['stop_reason'], 'baseline_gate_failed')

    def test_three_generations_have_balanced_groups_and_fixed_evaluations(self):
        manifest, calls, updates = self.run_campaign(True)
        self.assertEqual(len(calls), 40)
        self.assertEqual(len(updates), 3)
        self.assertTrue(manifest['complete'])
        for episodes in updates:
            self.assertEqual([ep['reward_group'] for ep in episodes], ['standing']*4 + ['moving']*4)
        baseline = [(c['scenario_id'], c['seed']) for c in calls if c['evaluation'] and c['generation'] == 0]
        for generation in (1, 2, 3):
            self.assertEqual([(c['scenario_id'], c['seed']) for c in calls
                              if c['evaluation'] and c['generation'] == generation], baseline)
        training_seeds = [c['seed'] for c in calls if not c['evaluation']]
        self.assertEqual(len(set(training_seeds)), 24)
        self.assertTrue(all(g['reload_verified'] for g in manifest['generations']))


@unittest.skipUnless(os.environ.get('THUNDERHILL_GODOT'), 'Native Godot required')
class NativeCollectionTests(unittest.TestCase):
    def test_moving_collection_metrics_use_policy_control_only(self):
        import numpy as np
        root = Path(tempfile.mkdtemp(prefix='diagnostic-collector-',
            dir=Path(__file__).resolve().parents[1] / 'artifacts'))

        class ProtocolFixture:
            def sample(self, **inputs):
                assert inputs['ego_history_xyz'][0, 0] < -1
                # Protocol fixture only. Never optimized or used as demonstration data.
                xyz = np.zeros((64, 3))
                xyz[:, 0] = np.arange(1, 65) * .5
                return dict(xyz=xyz, old_logprob=0., replay={'fixture': True})

        args = SimpleNamespace(godot=os.environ['THUNDERHILL_GODOT'], seconds=.5, rollouts=8)
        episode, result = collect(ProtocolFixture(), args, root / 'moving', generation=0,
            rollout=1, evaluation=True, digest='a'*64, seed=1073,
            initial_speed_m_s=5., scenario_id='moving')
        self.assertEqual(len(episode['replays']), 1)
        self.assertEqual(episode['reward_group'], 'moving')
        self.assertEqual(result['model_control_start_tick'], 180)
        self.assertEqual(result['scenario_setup_actions'], 15)
        self.assertAlmostEqual(result['reward_components']['sim_seconds'], .5)
        self.assertGreater(result['max_sampled_speed_m_s'], 1)
        self.assertEqual(len(result['trajectory_metrics']), 1)


if __name__ == '__main__':
    unittest.main()
