# Racing reward contract

The current reward uses NVIDIA AlpaGym's unchanged metric reward dispatcher.
The simulator supplies episode metrics; the native dispatcher applies their
configured scales and the native trainer computes group relative advantages.
This is gameplay reinforcement learning. No demonstration actions, supervised
imitation, or expert racing line enter the reward.

| Metric | Scale | Meaning |
| --- | --- | --- |
| progress | 1 | Signed legal distance divided by full track length, clipped to zero through one |
| collision_any | negative 10 | Any actual contact during the episode |
| offroad | negative 5 | Any offroad physics tick during the episode |
| fall_without_collision | negative 10 | Motorcycle fall without an already charged contact |
| completed_lap_speed | 1 | Safe completion bonus described below |

Progress, collision, and offroad retain the NVIDIA baseline scales. Falls and
the completion speed term are explicit motorcycle racing extensions. The full
track is our progress reference; NVIDIA's scene progress uses a recorded route.
We do not penalize distance from a presumed optimal racing line.

For a completed lap with no collision, offroad tick, or fall:

`completed_lap_speed = max(0, 1 - elapsed_sim_seconds / episode_budget_seconds)`

Every other outcome receives zero speed bonus. Duration excludes stationary
sensor warmup and uses simulator ticks, not inference wall time. The budget is
fixed across a rollout group. A faster valid lap receives a strictly greater
bonus while it finishes within the budget. The bonus is bounded below one and
cannot outweigh even the smallest safety penalty. A crash cannot evade an
accumulating time penalty because there is no such penalty. Idling earns no
speed bonus and no additional signed progress.

Unfinished attempts retain their native normalized progress reward. At equal
fixed budgets, travelling farther legally rewards useful speed; two unfinished
attempts with identical legal distance remain tied. This deliberately avoids
rewarding a short sprint followed by a crash or early voluntary termination.

The simulator verifies ordered track gates and lap validity. The bonus also
requires legal distance within five metres of the full circuit, matching the
maximum accepted displacement for a gate crossing in one physics tick. Safety flags are
accumulated over every executed physics tick, so a transient excursion cannot
be hidden by the final observation. Existing stalled attempt stops remain in
force. Reward versioning requires a freshly prepared run after a change.

## Primary implementation references

[NVIDIA metric dispatcher](https://github.com/NVlabs/AlpaGym/blob/972d160eed0e23d388497851504a3a233fec5879/packages/runtime/src/alpagym_runtime/rewards/compute.py)
provides the actual episode reward implementation used here. Local implementation:
[simulator metrics](../training/alpagym_metrics.py).

[Gymnasium CarRacing](https://github.com/Farama-Foundation/Gymnasium/blob/694e9124d734bbbfe25cd0cd550ff44077b14afc/gymnasium/envs/box2d/car_racing.py#L563)
provides a precedent for preferring fewer steps to finish. We use a terminal
completion bonus instead of its per step cost so premature failures cannot save
that cost. This reference is a design precedent, not evidence that our policy
has learned to race.

Reward arithmetic and recorded gameplay rescoring establish the intended
ordering. Only a real policy update followed by comparable gameplay can establish
learning improvement. Compare completion rate, crashes, legal distance, and
valid lap time alongside scalar reward; never infer faster racing from the
scalar alone.
