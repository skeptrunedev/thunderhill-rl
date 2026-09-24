"""One auditable generation of trajectory policy RL from real game rewards."""
import argparse
import json
import math
from pathlib import Path
import time

from driving_episode import DrivingEpisode
from experiment_tracking import ExperimentTracker
from lap_episode import REWARD_VERSION


MODEL = 'nvidia/Alpamayo-1.5-10B'
REVISION = '7aba8293c09993f2e125c6819df05d7fa3e873ea'
INSTRUCTION = ('Race forward along this paved racing circuit. Follow the track through its turns, '
               'stay inside the pavement boundaries, and make fast forward progress. '
               'You are riding a motorcycle. There is no traffic or speed limit.')


def publish(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def collect(policy, args, directory, *, generation, rollout, evaluation, digest, seed,
            initial_speed_m_s=0.0, scenario_id='standing'):
    started = time.monotonic()
    trajectory_metrics = []
    speeds = []
    with DrivingEpisode(godot=args.godot, output=directory, adapter_sha256=digest,
                        model=MODEL, revision=REVISION, generation=generation,
                        rollout=rollout, rollout_count=args.rollouts, evaluation=evaluation,
                        time_budget_seconds=args.seconds, initial_speed_m_s=initial_speed_m_s) as run:
        while not run.episode.done:
            observation = run.model_input()
            prediction = policy.sample(camera_frames=observation['images'],
                ego_history_xyz=observation['ego_history_xyz'],
                ego_history_rot=observation['ego_history_rot'],
                seed=seed + len(run.replays), instruction=INSTRUCTION)
            xyz = prediction['xyz']
            if xyz.shape != (64, 3):
                raise RuntimeError('Published action contract requires 64 XYZ waypoints at 10Hz')
            trajectory_metrics.append(dict(first_x_m=float(xyz[0, 0]),
                half_second_x_m=float(xyz[4, 0]), horizon_x_m=float(xyz[-1, 0])))
            run.execute(prediction, hold_steps=5)
            speeds.append(float(run.episode.observation['state']['speed']))
            print(json.dumps(dict(event='trajectory', generation=generation, rollout=rollout,
                evaluation=evaluation, tick=run.episode.observation['tick'],
                speed=run.episode.observation['state']['speed'],
                elapsed_seconds=time.monotonic() - started)), flush=True)
        summary = run.finish()
        summary.update(scenario_id=scenario_id, sample_seed=seed,
                       initial_speed_m_s=initial_speed_m_s, trajectory_metrics=trajectory_metrics,
                       mean_sampled_speed_m_s=sum(speeds) / len(speeds),
                       max_sampled_speed_m_s=max(speeds),
                       wall_seconds=time.monotonic() - started)
        controls = [row['controls'] for row in run.episode.records
                    if 'controls' in row and row.get('action_source') != 'scenario_setup']
        summary['max_policy_throttle'] = max(row['throttle'] for row in controls)
        summary['mean_policy_throttle'] = sum(row['throttle'] for row in controls) / len(controls)
        publish(directory / 'summary.json', summary)
        if not summary['training_eligible']:
            raise RuntimeError('Simulator attempt failed audit or collection eligibility')
        return dict(reward=summary['reward_components']['total'], replays=run.replays,
                    reward_group=scenario_id), summary


SCENARIOS = (dict(scenario_id='standing', initial_speed_m_s=0.0),
             dict(scenario_id='moving', initial_speed_m_s=5.0))
EVALUATION_SEEDS = (1073, 2073)
BASELINE_CRITERIA = dict(minimum_controlled_progress_m=20.0, minimum_final_speed_m_s=1.0,
                         minimum_first_plan_horizon_x_m=2.0)


def baseline_gate(summaries):
    """Conservative diagnostic gate, not a claim of learned racing capability."""
    if len(summaries) != 4 or any(sum(r['scenario_id'] == s['scenario_id']
            for r in summaries) != 2 for s in SCENARIOS):
        raise ValueError('Baseline requires two attempts per starting condition')
    results = []
    for row in summaries:
        if row['scenario_id'] != 'moving':
            continue
        reasons = []
        if not row['recording_provenance_verified'] or not row['training_eligible']:
            reasons.append('invalid_recording')
        if row['reward_components']['legal_progress_m'] < BASELINE_CRITERIA['minimum_controlled_progress_m']:
            reasons.append('insufficient_forward_progress')
        final = row['final_observation']
        if final['state']['speed'] < BASELINE_CRITERIA['minimum_final_speed_m_s']:
            reasons.append('did_not_maintain_motion')
        if final['state']['crashed'] or row['offtrack_ticks'] or row['reason'] in ('stalled', 'track_limits'):
            reasons.append('failed_to_follow_straight')
        if row['trajectory_metrics'][0]['horizon_x_m'] < BASELINE_CRITERIA['minimum_first_plan_horizon_x_m']:
            reasons.append('first_plan_near_stationary_or_reverse')
        results.append(dict(seed=row['sample_seed'], passed=not reasons, reasons=reasons))
    return dict(passed=all(r['passed'] for r in results), criteria=BASELINE_CRITERIA,
                moving_attempts=results)


def scenario_metrics(rows, prefix):
    result = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row['scenario_id'] == scenario['scenario_id']]
        if not selected:
            continue
        base = prefix + '/' + scenario['scenario_id'] + '/'
        for name, values in (
            ('reward', [r['reward_components']['total'] for r in selected]),
            ('progress_m', [r['reward_components']['legal_progress_m'] for r in selected]),
            ('final_speed_m_s', [r['final_observation']['state']['speed'] for r in selected]),
            ('stall_rate', [int(r['reason'] == 'stalled') for r in selected]),
            ('offtrack_ticks', [r['offtrack_ticks'] for r in selected]),
            ('max_policy_throttle', [r.get('max_policy_throttle', 0.) for r in selected]),
            ('mean_sampled_speed_m_s', [r.get('mean_sampled_speed_m_s', 0.) for r in selected]),
        ):
            result[base + name] = sum(values) / len(values)
    return result


def unused_attempt_path(path):
    """Keep every interrupted recording in place; collect into a new attempt."""
    candidate = path
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = path.with_name(path.name + f'-attempt-{attempt:03d}')
    return candidate


def resume_checkpoint(policy, args, manifest):
    """Restore the last committed policy and its Adam state, never partial updates."""
    if (manifest['model'] != MODEL or manifest['revision'] != REVISION
            or manifest['reward_version'] != REWARD_VERSION
            or manifest['rollouts_per_generation'] != args.rollouts
            or manifest['time_budget_seconds'] != args.seconds
            or manifest.get('supervised_training_performed') is not False):
        raise ValueError('Resume cannot change model, reward, or rollout contract')
    digest = manifest['initial_adapter_sha256']
    checkpoint = args.output / 'initial_adapter'
    for index, generation in enumerate(manifest['generations'], 1):
        if (generation['generation'] != index
                or generation['previous_adapter_sha256'] != digest
                or not generation['reload_verified']
                or generation['update']['optimizer_steps'] != 1):
            raise ValueError('Broken generation checkpoint lineage')
        digest = generation['adapter_sha256']
        checkpoint = args.output / generation.get('adapter_path', f'generation-{index:03d}/adapter')
    if args.generations < len(manifest['generations']):
        raise ValueError('Requested total cannot discard completed generations')
    policy.restore_training(checkpoint)
    if policy.fingerprint() != digest:
        raise RuntimeError('Resume checkpoint fingerprint mismatch')
    return digest, checkpoint


def run_diagnostic(policy, args, tracker, manifest, initial):
    """Four fixed evaluations and at most three generations of eight ownplays."""
    manifest_path = args.output / 'campaign.json'
    current = initial
    previous_path = args.output / 'initial_adapter'
    total = getattr(args, 'generations', 3)
    completed = len(manifest['generations'])
    if getattr(args, 'resume', False):
        current, previous_path = resume_checkpoint(policy, args, manifest)

    def evaluate(generation):
        rows = []
        for scenario in SCENARIOS:
            for replicate, seed in enumerate(EVALUATION_SEEDS):
                directory = args.output / f'evaluation-{generation:03d}' / f"{scenario['scenario_id']}-{replicate+1}"
                existing = [row for row in manifest['evaluations']
                    if row.get('generation') == generation and row['scenario_id'] == scenario['scenario_id']
                    and row['sample_seed'] == seed]
                if existing:
                    if len(existing) != 1 or existing[0]['adapter_sha256'] != current:
                        raise RuntimeError('Evaluation checkpoint identity mismatch')
                    rows.append(existing[0])
                    continue
                directory.parent.mkdir(exist_ok=True)
                directory = unused_attempt_path(directory)
                episode, summary = collect(policy, args, directory, generation=generation,
                    rollout=replicate+1, evaluation=True, digest=current, seed=seed, **scenario)
                del episode
                rows.append(summary)
                manifest['evaluations'].append(summary)
                publish(manifest_path, manifest)
        tracker.log({'generation': generation, **scenario_metrics(rows, 'eval')})
        return rows

    if not completed:
        baseline = evaluate(0)
        gate = baseline_gate(baseline)
        manifest['baseline_gate'] = gate
        publish(manifest_path, manifest)
        # Driving failures are valid ownplay. Only broken mechanics/provenance block RL.
        readiness = all(row['training_eligible'] and row['recording_provenance_verified'] for row in baseline)
        for row in baseline:
            if row['scenario_id'] == 'moving':
                metrics = row.get('setup_speed_metrics') or {}
                readiness = readiness and (
                    metrics.get('max_m_s', float('inf')) - metrics.get('min_m_s', 0.) <= .15
                    and abs(metrics.get('final_m_s', 0.) - row['initial_speed_m_s']) <= .15)
        manifest['training_readiness'] = dict(passed=bool(readiness),
            criteria='valid_recordings_and_verified_steady_motion_history',
            driving_success_required=False)
        publish(manifest_path, manifest)
        if not readiness:
            raise RuntimeError('Baseline infrastructure or moving history verification failed')
        tracker.log({'generation': 0, 'eval/baseline_gate_passed': int(gate['passed']),
                     'eval/training_readiness_passed': 1})
    else:
        if not manifest.get('training_readiness', {}).get('passed'):
            raise RuntimeError('Missing verified baseline infrastructure')
        evaluate(completed)
    for generation in range(completed + 1, total + 1):
        episodes, summaries = [], []
        generation_path = unused_attempt_path(args.output / f'generation-{generation:03d}')
        generation_path.mkdir()
        for index in range(8):
            scenario = SCENARIOS[index // 4]
            episode, summary = collect(policy, args, generation_path / f'rollout-{index+1:04d}',
                generation=generation-1, rollout=index+1, evaluation=False, digest=current,
                seed=100000 * generation + 1000 * index + 73, **scenario)
            episodes.append(episode)
            summaries.append(summary)
            publish(generation_path / 'collection.json', dict(generation=generation,
                adapter_sha256=current, rollouts=summaries))
        update = policy.update(episodes, max_decisions=math.ceil(args.seconds / .5))
        if update['optimizer_steps'] != 1 or update['parameter_delta_l1'] <= 0:
            raise RuntimeError('Generation did not produce a nonzero policy update')
        adapter_path = generation_path / 'adapter'
        updated = policy.save(adapter_path)
        if updated == current:
            raise RuntimeError('Updated adapter hash unchanged')
        policy.reload(previous_path)
        if policy.fingerprint() != current:
            raise RuntimeError('Previous adapter reload failed')
        policy.reload(adapter_path)
        if policy.fingerprint() != updated:
            raise RuntimeError('Updated adapter reload failed')
        manifest['generations'].append(dict(generation=generation, update=update,
            previous_adapter_sha256=current, adapter_sha256=updated, reload_verified=True,
            adapter_path=str(adapter_path.relative_to(args.output))))
        current, previous_path = updated, adapter_path
        del episodes
        publish(manifest_path, manifest)
        tracker.log({'generation': generation, **scenario_metrics(summaries, 'rollout'),
            **{'update/' + k: v for k, v in update.items() if isinstance(v, (int, float))}})
        evaluate(generation)
    manifest.update(complete=True, diagnostic_complete=True, stop_reason='generations_completed')
    publish(manifest_path, manifest)
    print(json.dumps(dict(event='diagnostic_complete', generations=total)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--processor-path', required=True)
    parser.add_argument('--godot', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--rollouts', type=int, default=2)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = parser.parse_args()
    if args.diagnostic:
        if args.rollouts != 8 or args.seconds != 30:
            parser.error('Diagnostic requires eight rollouts and 30 seconds')
    elif not 2 <= args.rollouts <= 4 or not 0 < args.seconds <= 12:
        parser.error('Validation budget requires 2 to 4 rollouts and at most 12 simulated seconds')
    if args.resume and not args.diagnostic:
        parser.error('Resume is supported for diagnostic campaigns only')
    if not 1 <= args.generations <= 5:
        parser.error('Generation budget must be between one and five')
    from alpamayo_policy import AlpamayoPolicy
    args.output.mkdir(parents=True, exist_ok=args.resume)
    policy = AlpamayoPolicy(checkpoint=args.checkpoint, processor_path=args.processor_path)
    initial = None if args.resume else policy.save(args.output / 'initial_adapter')
    manifest = dict(model=MODEL, revision=REVISION, generations_requested=1,
                    rollouts_per_generation=args.rollouts, time_budget_seconds=args.seconds,
                    reward_version=REWARD_VERSION, supervised_training_performed=False,
                    initialization='published_driving_checkpoint_with_fresh_expert_lora',
                    training_method='continuous_trajectory_gameplay_policy_gradient',
                    action_semantics='trajectory_prediction_with_fixed_motorcycle_controller',
                    native_language_tool_calls=False, initial_adapter_sha256=initial,
                    camera_configuration='one_front_camera_four_frames',
                    prehistory='stationary_padding_at_standing_reset',
                    complete=False, generations=[], evaluations=[])
    manifest_path = args.output / 'campaign.json'
    if args.diagnostic:
        manifest.update(generations_requested=args.generations, evaluation_rollouts=4,
            maximum_attempts=4 + 12 * args.generations, baseline_criteria=BASELINE_CRITERIA,
            scenarios=list(SCENARIOS), evaluation_seeds=list(EVALUATION_SEEDS),
            prehistory='stationary_padding_or_recorded_steady_speed_setup_excluded_from_reward',
            training_on_valid_failures=True,
            diagnostic_complete=False, reward_grouping='within_matching_start_condition')
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        # Validate and restore before changing the authoritative manifest.
        resume_checkpoint(policy, args, manifest)
        initial = manifest['initial_adapter_sha256']
        manifest.update(generations_requested=args.generations,
                        maximum_attempts=4 + 12 * args.generations,
                        complete=False, diagnostic_complete=False)
        manifest.pop('stop_reason', None)
    publish(manifest_path, manifest)
    with ExperimentTracker(args.output, manifest, mode=args.wandb_mode,
                           entity='skeptrune-org', name=args.output.parent.name, resume=args.resume) as tracker:
        if args.diagnostic:
            run_diagnostic(policy, args, tracker, manifest, initial)
            return
        baseline_episode, baseline = collect(policy, args, args.output / 'baseline', generation=0,
                              rollout=1, evaluation=True, digest=initial, seed=1073)
        del baseline_episode
        manifest['evaluations'].append(baseline)
        publish(manifest_path, manifest)
        episodes, summaries = [], []
        for index in range(args.rollouts):
            episode, summary = collect(policy, args, args.output / f'rollout-{index+1:04d}',
                generation=0, rollout=index+1, evaluation=False, digest=initial,
                seed=73 + 1000 * index)
            episodes.append(episode)
            summaries.append(summary)
            manifest['training_rollouts'] = summaries
            publish(manifest_path, manifest)
        update = policy.update(episodes, max_decisions=math.ceil(args.seconds / 0.5))
        if update['optimizer_steps'] != 1 or update['parameter_delta_l1'] <= 0:
            raise RuntimeError('Generation did not produce a real nonzero policy update')
        final = policy.save(args.output / 'adapter')
        if final == initial:
            raise RuntimeError('Saved adapter did not change')
        # Replace the live updated weights first, so a no-op reload cannot pass.
        policy.reload(args.output / 'initial_adapter')
        if policy.fingerprint() != initial:
            raise RuntimeError('Initial adapter reload did not restore original weights')
        policy.reload(args.output / 'adapter')
        if policy.fingerprint() != final:
            raise RuntimeError('Reloaded adapter does not match saved weights')
        del episodes
        manifest['generations'].append(dict(generation=1, update=update, adapter_sha256=final))
        manifest['reload_verification'] = dict(initial_adapter_sha256=initial,
            updated_adapter_sha256=final, restored_initial_then_updated=True)
        tracker.log({'generation': 1, **{'update/' + key: value for key, value in update.items()
                                      if isinstance(value, (float, int))},
                     'rollout/mean_reward': sum(x['reward_components']['total'] for x in summaries) / len(summaries)})
        publish(manifest_path, manifest)
        _, evaluation = collect(policy, args, args.output / 'evaluation', generation=1,
                                rollout=1, evaluation=True, digest=final, seed=1073)
        manifest['evaluations'].append(evaluation)
        tracker.log({'generation': 1, 'eval/reward': evaluation['reward_components']['total'],
                     'eval/baseline_reward': baseline['reward_components']['total'],
                     'eval/progress_m': evaluation['reward_components']['legal_progress_m']})
        manifest['complete'] = True
        publish(manifest_path, manifest)
        print(json.dumps(dict(event='generation_complete', update=update,
                              initial_adapter_sha256=initial, adapter_sha256=final)), flush=True)


if __name__ == '__main__':
    main()
