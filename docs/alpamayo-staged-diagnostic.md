# Alpamayo staged driving diagnostic

The authorized experiment is at most three generations with eight training
rollouts per generation. Four fixed evaluation attempts run before training and
after every generation, for at most 40 attempts. Each attempt allows 30 seconds
of model control, with existing stall and track limits stopping failures early.
Every attempt retains its recording, camera inputs, sampled trajectories,
diffusion replay and complete video queue. No supervised training is used.

Use the published Alpamayo 1.5 10B checkpoint with a fresh action expert adapter.
The earlier one update validation is not assumed to be an improved starting
policy. Sampling uses the existing official stochastic trajectory distribution.
The VLM remains frozen. Only the action expert adapter receives policy gradients.

There are two starting conditions at the track origin:

* Standing, using stationary history padding.
* Moving at 5 metres per second, followed by five seconds of recorded speed hold
  to capture sixteen steady poses and four camera frames before model control.

The speed hold prefix is explicit scenario setup. It is recorded and audited,
never used as a training target, and excluded from reward, model duration and
stall timing. An initial moving speed without actual history would incorrectly
present zero velocity to the driving model, which estimates speed from poses.

Each generation collects four rollouts per condition. Advantages are standardized
within the matching condition, so starting momentum cannot win against standing
starts. Evaluation uses seeds 1073 and 2073 for each condition at every generation.
Training seeds are separate and change each generation.

The baseline driving metrics (progress, final speed, stalls and predicted motion)
remain diagnostic, but do not block RL on valid failed attempts. Training requires
valid recordings, actual finite model trajectories, and verified steady motion
history. The final sixteen setup speeds must span at most 0.15 metres per second
and finish within 0.15 metres per second of the requested starting speed.

Perform three generations, each with one grouped update, saved adapter, verified
reload from previous to updated weights, and fixed evaluation. There is no
scripted driving after setup: the fixed controller follows only model trajectories.
W&B reports rewards, progress, speed and stall rate separately by start condition.

Launch using the existing CLI:

```sh
env -u HF_TOKEN modal run --detach training/modal_alpamayo_rl.py \
  --run-id RUN_ID --diagnostic
```

The flag selects the expanded diagnostic; the original bounded validation remains
available without it. One RTX PRO 6000 is capped at 5400 seconds of child execution
with no retries. Hardware camera preflight is included in that budget. The normal
HF_TOKEN environment is preserved because the credential override is scoped to
this subprocess. Model weights are prepared on CPU before allocating the GPU.

## Earlier diagnostic result (before controller repair)

Run `alpamayo-diagnostic-20260924T024801Z` completed all four baseline attempts and
stopped at the gate. No training generation or optimizer update was performed.
Child execution took 204.76 seconds. The app is stopped with no remaining tasks
or containers. All four complete videos passed hash and frame count verification.

| Scenario | Seed | Model controlled progress | Final speed | Outcome |
| :--- | ---: | ---: | ---: | :--- |
| Standing | 1073 | negative 0.143 m | 0.014 m/s | Stalled |
| Standing | 2073 | 0.094 m | 0.089 m/s | Stalled |
| Moving | 1073 | 3.723 m | 0.008 m/s | Stalled |
| Moving | 2073 | 3.441 m | 0.083 m/s | Stalled |

Each received 10 seconds of model control before the stall stop. Moving history
was real, but it was decelerating: the setup coast reduced speed from 5 to 2.680
metres per second before model control began. The first predicted segment speeds
then decreased from 2.630 to 2.162 metres per second for seed 1073, and from 2.604
to 1.987 for seed 2073, over five control steps. Executed speeds tracked these
slowing targets. The model did not request fast forward driving that the controller
subsequently failed to deliver.

This is a negative result for the present configuration, not a conclusive test of
the model's capabilities. The decelerating history is a confound: it may encourage
continuing to slow down. Source inspection also confirms that the fixed controller
has proportional speed feedback without compensation for engine braking and
rolling resistance. Before another paid training run, validate steady speed
history and constant speed trajectory tracking. Do not extrapolate these four
failed baselines into a claim that more generations would necessarily work or
that Alpamayo can never drive this simulator.

See [machine readable results](alpamayo-staged-diagnostic-result.json) and
[W&B](https://wandb.ai/skeptrune-org/thunderhill-rl/runs/y14b3mt3).
Recordings and videos are archived under
`artifacts/modal-alpamayo-diagnostic-20260924T024801Z/`.

## Controller repair and training eligibility

The original proportional speed controller requested zero throttle at exact
requested speed, so engine braking and drag necessarily slowed the bike. PI
feedback now compensates using measured speed error, preserves integral state
across replans, prevents saturation windup, and clears integral for stopped or
reverse plans. It receives no track geometry or demonstration actions.

Actual Godot tests measured 4.9925 metres per second at five seconds and 5.0033 at
ten seconds for a constant 5 metre per second trajectory. The final sixteen setup
samples varied by only 0.0362 metres per second. Acceleration and stopping tests
also pass. Full rendered collection verifies steady camera history, exclusion of
setup reward, and audit reconstruction of every setup control from recorded state.

After the explicit request to fix the environment and start training, valid
stalls are retained as RL experience. Poor driving is no longer a prerequisite
for stopping the campaign. Nonfinite outputs, invalid recordings, bad setup
history, inconsistent behavior probabilities and zero optimizer updates remain
fatal. The earlier failed run and its evidence are preserved unchanged.

## Recovery and five generation continuation

The September 24 campaign completed two optimizer updates before Modal preempted
its GPU during the final generation 2 evaluation. Automatic restart failed because
the launcher required a new directory. Both adapter checkpoints and Adam optimizer
states survived in the Modal volume.

Diagnostic campaigns now accept `--generations 5 --resume`. Recovery verifies the
committed checkpoint lineage, restores weights and optimizer moments, preserves
completed evaluations, and finishes the missing evaluation before new gameplay.
Interrupted attempts remain in place; replacement attempts use distinct directory
names. An uncommitted generation is recollected from the last committed policy,
not represented as a completed update. Every explicit launch retains source
provenance, separate invocation logs, and a 5400 second deadline shared across
Modal preemption restarts. Tracking resumes the same W&B run.

The five generation target is 40 training rollouts and 24 fixed evaluations,
excluding additional interrupted attempts. This changes the experiment duration,
not the reward, starting conditions, or training method. Completion still requires
five actual updates, checkpoint reload verification, and all fixed evaluations.
