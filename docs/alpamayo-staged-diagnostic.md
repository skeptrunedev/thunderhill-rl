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
