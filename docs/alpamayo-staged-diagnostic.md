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
* Moving at 5 metres per second, followed by 1.5 seconds of real neutral coasting
  to capture sixteen poses and four camera frames before model control.

The coasting prefix is explicit scenario setup. It is recorded and audited,
never used as a training target, and excluded from reward, model duration and
stall timing. An initial moving speed without actual history would incorrectly
present zero velocity to the driving model, which estimates speed from poses.

Each generation collects four rollouts per condition. Advantages are standardized
within the matching condition, so starting momentum cannot win against standing
starts. Evaluation uses seeds 1073 and 2073 for each condition at every generation.
Training seeds are separate and change each generation.

The baseline gate requires both moving attempts to produce at least 20 metres
of legal progress during model control, finish at 1 metre per second or faster,
avoid stalls, crashes and leaving the track, and initially predict a forward
trajectory spanning at least 2 metres. These are conservative diagnostic cutoffs,
not empirical proof of learning or eventual lap completion. Standing starts may
fail without blocking the moving condition investigation. Log throttle and speed
alongside progress so coasting is not confused with purposeful acceleration.

If the gate fails, archive all four attempts and stop without an optimizer update.
If it passes, perform three generations, each with one grouped update, saved
adapter, verified reload from previous to updated weights, and fixed evaluation.
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

## Observed result

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
