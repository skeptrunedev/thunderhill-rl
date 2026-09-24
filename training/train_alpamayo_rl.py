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
               'You are starting a motorcycle from rest. There is no traffic or speed limit.')


def publish(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def collect(policy, args, directory, *, generation, rollout, evaluation, digest, seed):
    started = time.monotonic()
    with DrivingEpisode(godot=args.godot, output=directory, adapter_sha256=digest,
                        model=MODEL, revision=REVISION, generation=generation,
                        rollout=rollout, rollout_count=args.rollouts, evaluation=evaluation,
                        time_budget_seconds=args.seconds) as run:
        while not run.episode.done:
            observation = run.model_input()
            prediction = policy.sample(camera_frames=observation['images'],
                ego_history_xyz=observation['ego_history_xyz'],
                ego_history_rot=observation['ego_history_rot'],
                seed=seed + len(run.replays), instruction=INSTRUCTION)
            run.execute(prediction, hold_steps=5)
            print(json.dumps(dict(event='trajectory', generation=generation, rollout=rollout,
                evaluation=evaluation, tick=run.episode.observation['tick'],
                speed=run.episode.observation['state']['speed'],
                elapsed_seconds=time.monotonic() - started)), flush=True)
        summary = run.finish()
        if not summary['training_eligible']:
            raise RuntimeError('Simulator attempt failed audit or collection eligibility')
        return dict(reward=summary['reward_components']['total'], replays=run.replays), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--processor-path', required=True)
    parser.add_argument('--godot', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--rollouts', type=int, default=2)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    args = parser.parse_args()
    if not 2 <= args.rollouts <= 4 or not 0 < args.seconds <= 12:
        parser.error('Validation budget requires 2 to 4 rollouts and at most 12 simulated seconds')
    from alpamayo_policy import AlpamayoPolicy
    args.output.mkdir(parents=True, exist_ok=False)
    policy = AlpamayoPolicy(checkpoint=args.checkpoint, processor_path=args.processor_path)
    initial = policy.save(args.output / 'initial_adapter')
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
    publish(manifest_path, manifest)
    with ExperimentTracker(args.output, manifest, mode=args.wandb_mode,
                           entity='skeptrune-org', name=args.output.parent.name) as tracker:
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
        policy.reload(args.output / 'adapter')
        if policy.fingerprint() != final:
            raise RuntimeError('Reloaded adapter does not match saved weights')
        del episodes
        manifest['generations'].append(dict(generation=1, update=update, adapter_sha256=final))
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
