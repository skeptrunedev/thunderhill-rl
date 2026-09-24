"""Camera and trajectory collection on the existing audited simulator lifecycle.

The trajectory is the sampled model action. Serialized control_bike commands
below are deterministic controller outputs, not sampled language model tokens.
"""
from collections import deque
import base64
import hashlib
import io
import json

import numpy as np
import torch
from PIL import Image

from lap_episode import LapEpisode
from lap_policy import RoadTelemetry
from check_parallel import worker
from driving_trajectory import (
    TrajectoryTracker, godot_history_to_ego, encode_controller_controls,
    decode_controller_controls,
)


def rendered_worker(godot, directory, timeout, extra_args=()):
    return worker(godot, directory, timeout, extra_args, rendered=True)


class DrivingEpisode:
    def __init__(self, **kwargs):
        self.initial_speed_m_s = kwargs.get("initial_speed_m_s", 0.0)
        self.episode = LapEpisode(
            scenario_setup_ticks=600 if self.initial_speed_m_s > 0 else 0, scenario_setup_kind="speed_hold",
            road=RoadTelemetry(), worker_factory=rendered_worker,
            action_parser=decode_controller_controls, **kwargs)
        self.images = deque(maxlen=4)
        self.positions = deque(maxlen=16)
        self.headings = deque(maxlen=16)
        self.image_ids = deque(maxlen=4)
        self.tracker = TrajectoryTracker()
        self.replays = []

    def __enter__(self):
        self.episode.__enter__()
        try:
            self.plans = (self.episode.output / 'trajectory_decisions.jsonl').open('w')
            self.capture(pad=self.initial_speed_m_s == 0)
            if self.initial_speed_m_s > 0:
                # Recorded scenario initialization only, excluded from policy replay/reward.
                speeds = deque(maxlen=16)
                speeds.append(self.episode.observation['state']['speed'])
                for _ in range(self.episode.scenario_setup_ticks // 12):
                    self.tracker.replan([[(i + 1) * .1 * self.initial_speed_m_s, 0., 0.] for i in range(64)])
                    state = self.episode.observation['state']
                    controls, _ = self.tracker.next_controls(state['speed'], state['lean'])
                    self.episode.apply(encode_controller_controls(controls), [], [], scenario_setup=True)
                    self.capture(pad=False)
                    speeds.append(self.episode.observation['state']['speed'])
                    if self.episode.done:
                        raise RuntimeError("Moving scenario terminated during recorded history setup")
                self.setup_speed_metrics = dict(min_m_s=min(speeds), max_m_s=max(speeds),
                                               final_m_s=speeds[-1], target_m_s=self.initial_speed_m_s)
                if max(speeds) - min(speeds) > .15 or abs(speeds[-1] - self.initial_speed_m_s) > .15:
                    raise RuntimeError(f"Moving history failed steady speed check: {self.setup_speed_metrics}")
                self.episode.begin_model_control()
        except BaseException as error:
            import sys
            try:
                self.episode.__exit__(*sys.exc_info())
            except BaseException as cleanup_error:
                error.add_note(f'Episode cleanup also failed: {cleanup_error!r}')
            if hasattr(self, 'plans'):
                self.plans.close()
            raise
        return self

    def __exit__(self, *error):
        try:
            return self.episode.__exit__(*error)
        except BaseException as cleanup_error:
            if error[1] is None:
                raise
            error[1].add_note(f'Episode cleanup also failed: {cleanup_error!r}')
            return False
        finally:
            self.plans.close()

    def capture(self, *, pad=True):
        observation = self.episode.observation
        response = self.episode.client.request(dict(op='capture',
            episode_id=self.episode.episode_id, expected_tick=observation['tick']))
        if ('error' in response or response.get('tick') != observation['tick']
                or response.get('episode_id') != self.episode.episode_id):
            raise RuntimeError(f'Invalid camera receipt: {response.get("error", "identity mismatch")}')
        raw = base64.b64decode(response['image']['base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != response['image']['sha256']:
            raise ValueError('Camera pixels differ from recorded observation')
        pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB')).copy()
        self.images.append(pixels)
        self.image_ids.append(response['observation_id'])
        self.positions.append(list(observation['state']['position']))
        self.headings.append(observation['state']['heading'])
        # At a standing reset the unavailable prehistory is stationary padding.
        while pad and len(self.images) < 4:
            self.images.appendleft(pixels.copy())
            self.image_ids.appendleft(response['observation_id'])
        while pad and len(self.positions) < 16:
            self.positions.appendleft(self.positions[0][:])
            self.headings.appendleft(self.headings[0])

    def model_input(self):
        xyz, rotation = godot_history_to_ego(list(self.positions), list(self.headings))
        return dict(images=np.stack(self.images), ego_history_xyz=np.asarray(xyz),
                    ego_history_rot=np.asarray(rotation), tick=self.episode.observation['tick'])

    def execute(self, prediction, *, hold_steps=5):
        if self.episode.done:
            raise ValueError('Cannot execute a plan after episode completion')
        plan = np.asarray(prediction['xyz'], dtype=np.float64)
        self.tracker.replan(plan)
        plan_index = len(self.replays)
        replay_directory = self.episode.output / 'trajectory_replays'
        replay_directory.mkdir(exist_ok=True)
        replay_path = replay_directory / f'{plan_index:04d}.pt'
        pending = replay_path.with_suffix('.pending')
        torch.save(prediction['replay'], pending)
        pending.replace(replay_path)
        self.replays.append(prediction['replay'])
        start_tick = self.episode.observation['tick']
        record = dict(plan_index=plan_index, before_tick=start_tick,
                      camera_observation_ids=list(self.image_ids), xyz=plan.tolist(),
                      replay_path=str(replay_path.relative_to(self.episode.output)),
                      replay_sha256=hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                      old_logprob=float(prediction['old_logprob']),
                      action_semantics='sampled_trajectory_with_fixed_motorcycle_controller',
                      controls=[])
        try:
            for _ in range(hold_steps):
                state = self.episode.observation['state']
                controls, diagnostics = self.tracker.next_controls(state['speed'], state['lean'])
                command = encode_controller_controls(controls)
                row = self.episode.apply(command, [], [])
                record['controls'].append(dict(tick=row.get('tick'), controls=row.get('controls'),
                                               diagnostics=diagnostics))
                if self.episode.done:
                    break
                self.capture()
        finally:
            record['after_tick'] = self.episode.observation['tick']
            self.plans.write(json.dumps(record, allow_nan=False) + '\n')
            self.plans.flush()

    def finish(self):
        summary = self.episode.finish()
        summary['action_semantics'] = 'sampled_trajectory_with_fixed_motorcycle_controller'
        summary['trajectory_decisions'] = len(self.replays)
        summary['setup_speed_metrics'] = getattr(self, 'setup_speed_metrics', None)
        (self.episode.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
