# Racing reward references

Source audit on September 22, 2026. These are inspected implementations, not independently reproduced training results. They establish precedents for reward design, not proof that the same weights will train a language model to ride a motorcycle.

## TMRL, Trackmania

The reward helper computes `(next_reference_index - previous_reference_index) / 100`. The environment then adds a configurable constant reward each step and a configurable finish bonus. It terminates at the finish or after sufficiently many steps without forward progress, following an initial grace period. Reward is suppressed when the position is too far from the reference trajectory.

The index is a reference trajectory sample index, not necessarily meters. The helper computes reward before its backward index search, so this is not a clean signed distance reward. Do not copy that detail into our implementation. Its reference trajectory is used to measure progress; our simulator already has track geometry and does not need demonstration actions.

Sources: [reward helper, lines 14 to 115](https://github.com/trackmania-rl/tmrl/blob/10266a7d2351e727f51ab302b3dd6592895d1763/tmrl/custom/tm/utils/compute_reward.py#L14), [environment reward additions, lines 185 to 210](https://github.com/trackmania-rl/tmrl/blob/10266a7d2351e727f51ab302b3dd6592895d1763/tmrl/custom/tm/tm_gym_interfaces.py#L185).

Lesson: reward forward track progress directly, recognize the finish, and consider ending stalled episodes. The numeric divisor 100 does not establish the right scale for our meters.

## Gymnasium CarRacing

Each newly visited road tile earns `1000 / number_of_track_tiles`; each action frame costs `0.1`. A completed lap terminates the episode. Going outside the playfield terminates with a reward of negative 100 for that step. This boundary is not the same as merely leaving the pavement. Repeated visits to a tile do not earn the tile reward again.

Sources: [first visit reward, lines 89 to 100](https://github.com/Farama-Foundation/Gymnasium/blob/694e9124d734bbbfe25cd0cd550ff44077b14afc/gymnasium/envs/box2d/car_racing.py#L89), [step reward and termination, lines 563 to 586](https://github.com/Farama-Foundation/Gymnasium/blob/694e9124d734bbbfe25cd0cd550ff44077b14afc/gymnasium/envs/box2d/car_racing.py#L563).

Lesson: coverage earns reward, repeated coverage does not, and fewer frames to finish means a better score. A time cost must be calibrated against failure termination to avoid incentivizing an early crash.

## MetaDrive

Default dense reward is signed lane distance change plus `0.1 * speed / maximum_speed`, with a road direction factor. An optional lateral factor scales the distance reward and is disabled by default. At destination, the step reward is replaced with positive 10. Leaving the road or crashing into a vehicle or object replaces the step reward with negative 5. These failures terminate by default; termination is configurable. Earlier rewards remain accumulated.

Sources: [defaults, lines 70 to 94](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/envs/metadrive_env.py#L70), [reward function, lines 246 to 290](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/envs/metadrive_env.py#L246), [termination function, lines 133 to 203](https://github.com/metadriverse/metadrive/blob/85e5dadc6c7436d324348f6e3d8f8e680c06b4db/metadrive/envs/metadrive_env.py#L133).

Lesson: failure does not erase all previous progress. Raw speed can supplement progress, but legal progress is the safer initial signal for our track because speed alone does not establish useful movement.

## gym_torcs

The inspected repository uses `longitudinal_speed * cos(track_heading_error)` each step. An increase in damage replaces the step reward with negative 1. Leaving the track also gives negative 1 and ends the episode. After step 500, forward projected speed below 5 ends the episode. Driving backward also ends it. Damage alone does not trigger termination in this wrapper.

Source: [reward and termination, lines 132 to 166](https://github.com/ugo-nama-kun/gym_torcs/blob/da5d6ddec3a35718fea89dc1c05037743173c668/gym_torcs.py#L132).

Lesson: reward speed in the useful track direction rather than throttle or speed magnitude. This older implementation is a design reference, not a recommendation to adopt its weights or simulator wrapper.

## Thunderhill experiment

The approved experiment uses signed legal progress in meters divided by 100, a failure penalty of 0.2 for crashing or exceeding track limits, and the existing valid completion and completion speed bonuses. Training remains exclusively RL from the model's own native tool calls and resulting simulator transitions. No supervised imitation is authorized.

The penalty represents 20 meters of progress. A legal 40 meter attempt ending in a crash can therefore beat staying stationary, while clean driving over the same distance scores higher. This is a deliberate exploration tradeoff, not a claim that crashes are acceptable final behavior.

At a fixed episode budget, more legal distance rewards higher average useful speed. It does not distinguish two unfinished attempts with the same total legal distance but different movement timing. Completed lap time remains a separate audited metric and receives the existing completion speed bonus.

The current group relative trainer subtracts the group mean. A constant episode penalty shared by every rollout cancels. A penalty proportional to actual elapsed time does not necessarily cancel because failures can end episodes early, but it can favor early failure if poorly weighted. We are not adding that extra incentive in this experiment.

Evaluate legal distance, crashes, completion rate and valid lap times alongside reward. Higher reward alone is not evidence of better racing.

## Stalled attempt cutoff

New collector processes allow five seconds to launch, then require at least one meter of signed legal progress in each rolling five second window. The first possible stall is at ten simulated seconds. Reverse movement and retracing do not count as new progress. The cutoff uses physics ticks, not wall clock inference time. Simulator termination, invalid track limits, and the configured episode budget take precedence.

A stalled attempt remains eligible for RL, retains its earned progress reward, and is logged separately from a crash. Its recording and video job include the stop reason, window bounds, measured progress, and cutoff configuration. Native tool checkpoint evaluation uses the same collector. Processes already running retain the rules they loaded at launch.
