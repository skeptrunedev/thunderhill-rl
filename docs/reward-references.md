# Racing reward contract

The reward uses NVIDIA AlpaGym's unchanged metric reward dispatcher: one scalar
per rollout, the sum of configured scales times simulator metrics. The native
trainer computes group relative advantages and replays each rollout's advantage
on all of its valid steps. This is gameplay reinforcement learning. No
demonstration actions, supervised imitation, or expert racing line enter the
reward.

## Terms

Every term shares one unit: metres divided by the reference distance
`20 m/s * horizon_seconds`, the distance a rider averaging 20 m/s covers in the
fixed episode horizon (`episode_seconds`, warmup excluded).

| Metric | Scale | Meaning |
| --- | --- | --- |
| progress | 1 | Credited legal progress over the fixed horizon, in reference distances |

v5 removed v4's kinetic incident costs (`collision_cost`, `offroad_cost`,
`fall_cost`, still reported as diagnostics). Every incident already ends the
episode and forfeits the rest of the horizon, so charging it again made the v4
qualification rank a stall before a corner above attempting the corner and
running wide.

`progress` keeps NVIDIA's name, but it is a progress rate over a fixed horizon,
not a fraction of the full circuit. Credited progress is signed legal distance
since model control began, bounded by one lap. Horizon time that was not
executed because the attempt ended early (stall, crash, track limits, driver
termination) earns zero progress. A safe completed lap (ordered gates, full
distance, no contact, offroad tick or fall) is credited at its own mean lap
speed for the whole horizon, `track_length * horizon / lap_time`, so a faster
lap is always worth strictly more and no separate completion bonus exists.

`v` is the larger of the speed on the incident tick and the previous tick,
because obstacle contact zeroes the simulator speed on the contact tick. Each
incident is charged once at its onset. `collision_any`, `offroad` and
`fall_without_collision` remain in the aggregated metrics as 0/1 event flags
with NVIDIA's names for charts and evaluations; they are not reward terms.

Every rollout summary records `credited_progress_m`,
`horizon_progress_rate_m_s`, the incident list with speed and cost, and the
reward version. Changing the reward requires a freshly prepared run.

## Why this shape

GRPO divides each group's rewards by their standard deviation, so the overall
unit sets readable magnitudes, not behaviour. What matters is the relative value
of distinct outcomes. The previous contract charged a fixed 5 (offroad) or 10
(collision, fall) against progress measured in full laps, so one slow offroad
tick cost as much as five laps. Standing still was then the dominant strategy,
exactly the failure Fuchs et al. report for fixed wall penalties in GT Sport
("full braking and standing still"). Their fix, which we follow, scales the
penalty with kinetic energy: `r_t = progress_t - c_w |v_t|^2` during wall contact.

With the incident cost expressed in metres, braking to a stop at 0.25 g before
the incident is roughly distance-neutral against crashing, while a crash also
forfeits the rest of the horizon. Consequences, at a 30 s horizon:

* A stalled rider earns only the metres it covered, the minimum available
  without reversing or crashing.
* Any rider that keeps moving beats one that stops. A slow crash after real
  progress beats stalling, because 5 m/s costs only 5 m.
* A fast crash costs tens of metres (13 m at 8 m/s, 52 m at 16 m/s, 82 m at
  20 m/s) on top of the forfeited horizon.

## Scripted drive check (2026-09-25)

These are scripted evaluation drives through the real `GodotRuntime` path: a
gRPC driver follows the submitted route at a fixed speed, with an 8 m/s rolling
start and a 30 s horizon. They are not training data. v3 is the previous reward,
v4 the current one.

| Drive | End | Seconds | Legal m | v3 | v4 |
| --- | --- | --- | --- | --- | --- |
| stop | stalled | 10.0 | 1.7 | 0.0004 | 0.003 |
| 8 m/s, stop at 4 s | stalled | 10.0 | 35.8 | 0.0078 | 0.060 |
| 2 m/s | horizon | 30.0 | 60.0 | 0.0130 | 0.100 |
| 5 m/s | fall at 5.0 m/s | 18.0 | 90.6 | -9.980 | 0.143 |
| 8 m/s | fall at 8.3 m/s | 14.3 | 115.3 | -9.975 | 0.169 |
| 16 m/s | fall at 16.0 m/s | 10.2 | 156.6 | -9.966 | 0.174 |
| 12 m/s | fall at 12.2 m/s | 12.4 | 147.1 | -9.968 | 0.195 |
| 12 m/s straight | horizon | 30.0 | 356.2 | 0.077 | 0.594 |

Under v3 every drive that fell scored about 10 below standing still. Under v4,
standing still is worst, and the 16 m/s fall ranks below the 12 m/s fall despite
9.5 m more progress. On the 128 recorded campaign rollouts, all of which braked
to a stall within 5.9 to 7.6 m with no incident, v4 is an exact positive
rescaling of v3, so GRPO advantages are unchanged. When siblings behave
identically, no per-rollout reward can separate them.

## Stall termination

The stall monitor (less than 1 m of legal progress in a 5 s window after 5 s
grace) still ends the attempt. The reward no longer depends on when that
happens. Because the unexecuted horizon earns zero progress, a stalled attempt
scores exactly what the same bike would score if it stayed stopped for the full
horizon, so terminating early does not bias comparisons against full-length
siblings. Running out a stopped bike would spend a model inference every
0.2 simulated seconds (up to 3000 per 600 s rollout) to produce padding-free
steps carrying a known, near-minimal reward. The remaining cost of early
termination is fewer valid steps for that rollout in the replay pool. That is
acceptable because every one of its steps already carries its negative
advantage.

## Per-step credit (not implemented)

AlpaGym replays one advantage per rollout across every valid step
(`AlpagymGRPOTrainer._prepare_training_data` in
`packages/runtime/src/alpagym_runtime/cosmos/trainer.py`). A rollout that rides
well for 20 s and then crashes therefore pushes its good early decisions down
with its crash. Per-step reward-to-go would fix this, following the per-step
progress rewards in Fuchs et al. and GT Sophy. The protocol already has room
for it. `RolloutReturn.timestep_metrics` (repeated `TimestepMetric`: name,
timestamps_us, values, valid) could carry per control step progress and incident
costs from the bridge. `EpisodeMetrics.dense` in `alpagym_runtime/types.py` is
already persisted by the disk transport. The upstream changes would be:
`episode_runner/streaming_worker.py` (which currently builds
`EpisodeMetrics(aggregated=..., dense={})`) copies `timestep_metrics` into
`dense`, and `_prepare_training_data` uses a group-normalized per step
reward-to-go instead of `float(rollout.advantage)`. This changes NVIDIA's
worker and trainer, so it is a recommendation only.

## Primary implementation references

[NVIDIA metric dispatcher](https://github.com/NVlabs/AlpaGym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/rewards/compute.py)
provides the episode reward implementation used here. Local implementation:
[simulator metrics](../training/alpagym_metrics.py).

[Fuchs et al., Super-Human Performance in Gran Turismo Sport Using Deep RL](https://arxiv.org/abs/2008.07971),
Section III-A: centerline progress reward, plus a kinetic-energy wall contact
penalty (`c_w = 5e-4`), and their observation that fixed penalties produced
braking and standing still.

Reward arithmetic and scripted-drive rescoring establish the intended ordering.
Only a real policy update followed by comparable gameplay can establish learning
improvement. Compare completion rate, crashes, legal distance, and valid lap time
alongside the scalar reward. Never infer faster racing from the scalar alone.
