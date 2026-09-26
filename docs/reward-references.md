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
| progress | 1 | Centering weighted legal progress over the fixed horizon |
| termination_cost | -1 | Kinetic cost of the incident or stall that ended the episode |

`progress` (v6) sums, over every physics tick since model control began, the
tick's legal progress times `max(0, 1 - |lateral_m| / half_width_m)`, the
gym-donkeycar centering weight (`half_width_m` is the game's interpolated track
half width, the same one its on_track test uses). A ride that drifts to the
edge before leaving the track earns less than one that holds its line to the
same point. It keeps NVIDIA's name, but it is a rate over a fixed horizon, not
a fraction of the circuit, bounded by one lap. Horizon time not executed
because the attempt ended early earns zero. A safe completed lap is credited at
its own centered pace for the whole horizon, `centered * horizon / lap_time`.

`termination_cost` charges the episode's first incident (offroad, obstacle
contact or fall) once at onset, `v^2 / (2 a)`, with `a = 0.48 g`, the game's
`offtrack_friction` (godot/scripts/motorcycle.gd): the gravel run-off a rider
needs to stop at that surface's friction limit. `v` is the larger of the speed
on the incident tick and the previous tick, because obstacle contact zeroes the
simulator speed on the contact tick. A stall is charged as an offroad at
10 m/s (10.6 m). `collision_cost`, `offroad_cost`, `fall_cost`,
`collision_any`, `offroad` and `fall_without_collision` remain diagnostics under
NVIDIA's names.

Every rollout summary records `centered_progress_m`, `credited_progress_m`,
`termination`, `termination_cost_m`, the incident list and the reward version.
Changing the reward requires a freshly prepared run.

## Why this shape

GRPO divides each group's rewards by their standard deviation, so the overall
unit sets readable magnitudes, not behaviour. What matters is the relative value
of distinct outcomes inside one group of six rides from the same start.

v5 rewarded progress only. In a group where every ride left the track at the
same corner, the only difference was how far each got before leaving, and the
rides that carried more speed got further, so the policy got faster and stopped
turning. v4 had the opposite fault: a 0.25 g kinetic cost on every incident but
none on a stall, so braking to a stop before a corner beat attempting it. v6
follows Fuchs et al. (GT Sport) and GT Sophy, which scale off-track and wall
penalties with speed, and fixes v4 by pricing a stall.

Derivation of the constants, with `v` in m/s and costs in metres:

* Shape `v^2`. Kinetic energy is what a crash dissipates, and the convex cost
  is cheap at a modest pace (1.0 m at 3 m/s, 2.7 m at 5 m/s) and steep where
  the observed error is (24 m at 15 m/s, 42 m at 20 m/s, 77 m at 27 m/s).
  Written as `v * tau` with `tau = v / (2 a)`, it erases 0.3 s of riding at
  3 m/s, 1.6 s at 15 m/s and 2.1 s at 20 m/s.
* Arriving slower ranks higher. Two rides that leave at the same point share
  their progress, so any increasing cost orders them by speed. Where the faster
  ride also got further, it still ranks lower unless the extra centered progress
  exceeds `(v1^2 - v2^2) / (2 a)`: 15 m for 18.9 against 14.7 m/s.
* Crawling never beats a sensible pace. A rider at `v_s` that leaves the track
  after `t` seconds beats a crawler at `v_c` surviving the whole horizon `H`
  (same line) when `t > tau(v_s) + (v_c / v_s) H`. At 15 against 3 m/s over
  30 s that is 7.6 s instead of v5's 6.0 s; the cost moves the break-even by
  1.6 s, well below the `(v_s - v_c) H = 360 m` the crawler forfeits.
* Stalling is not free. Stopping before a corner is priced like leaving the
  track at 10 m/s, half the reference speed. At equal progress, any attempt
  that runs wide below 10 m/s ranks above stopping, and a faster attempt must
  have gained `(v^2 - 100) / (2 a)` more centered metres, 13 m at 15 m/s.

Sensitivity on the nine live-campaign groups below: at `a` = 0.25, 0.35, 0.48,
0.75 and 1.0 g the pooled correlation between onset speed (relative to the group
mean) and advantage is -0.36, -0.21, +0.02, +0.34 and +0.40 (v5: +0.52). The
friction-derived 0.48 g removes v5's reward for speed without v4's size.

## Recorded ride rescoring (2026-09-26)

The per-tick transitions recorded by the game for campaign
`alpamayo15-remote-godot-12h-20260926t0020` (rolling 15 m/s, 30 s horizon) were
replayed through v5 and v6. The v5 replay reproduces every recorded training
reward exactly. Older recordings lack `half_width_m`, so it was rebuilt from
`godot/data/track.json`; it agrees with the game's on_track flag on all 64,779
ticks. Spearman correlation between onset speed and reward among incident rides:

| Generation | Group | v5 | v6 |
| --- | --- | --- | --- |
| 4 | all offroad within 2.9 to 5.0 s, 14.7 to 18.9 m/s | +0.31 | -0.89 |
| 7 | all offroad within 2.7 to 3.3 s, 17.6 to 19.1 m/s | +0.31 | -0.26 |
| 8 | all offroad within 3.8 to 5.7 s, 16.3 to 19.1 m/s | +0.71 | +0.26 |
| 0 to 8 pooled | 51 incident rides, speed relative to group mean | +0.52 | +0.02 |

In generation 4, v5 ranked the 18.9 m/s ride second; v6 ranks it fifth and the
14.7 m/s ride first. In generation 1 (crawling at about 3 m/s) the stalled ride
stays last and its advantage falls from -1.30 to -2.05. Where a faster ride got
much further (generation 6: 119 m against 65 to 92 m) it still ranks first,
because no sibling slowed down. The qualification ride from each start
(`recordings/baseline`) outranks every learned ride at generations 4, 6 and 7
under both versions.

Scripted v6 drives through the real game (evaluation fixtures, 15 m/s rolling
start, 30 s horizon, start line and the generation 4 start) confirm the game
emits `half_width_m` and the stall pricing: standing still scores 0.009 (v5
0.028), braking to a stall after 1.5 s scores 0.043, a 3 m/s crawl 0.170 and a
route-following 12 m/s ride 0.601. Holding the initial heading with no route
(12 m/s) survives 30 s but drifts wide, so centering cuts it from 0.604 to
0.462. None of these scripted speeds left the track, so incident ordering rests
on the recorded rides above.

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
grace) still ends the attempt. The reward does not depend on when that
happens. Because the unexecuted horizon earns zero progress, a stalled attempt
earns exactly the progress the same bike would earn if it stayed stopped for
the full horizon, so terminating early does not bias comparisons against
full-length siblings. The stall is then charged its fixed `termination_cost`. Running out a stopped bike would spend a model inference every
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
