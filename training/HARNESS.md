# Bike agent harness

`ThunderhillEnv` in `agent_harness.py` is the initial TRL environment adapter.
It exposes rider inputs, not simulator administration. Each environment instance
owns a separate Godot process and user data directory, provisioned by the trainer.

## Agent tools

| Tool | Inputs | Output |
| --- | --- | --- |
| `observe()` | None | Last authoritative telemetry and current observation token, without advancing time |
| `control_bike(...)` | Current observation token, throttle, steering, front brake, rear brake, gear change | Telemetry after the fixed action interval and a fresh observation token |

Throttle and brake fractions lie between zero and one. Steering lies between
minus one and one. Gear change is minus one, zero or one. Steering remains the
simulator's assisted input, not validated physical handlebar torque. Automatic
shifting and assistance are fixed by the environment. The agent cannot change
those settings, choose extra physics ticks, reset its episode, set its reward,
or select the checkpoint identity.

Every control call advances the server's fixed 12 ticks, or 0.1 simulated seconds,
and may stop sooner at an episode boundary. A token from the latest observation
is mandatory. Successful advancement replaces it with a fresh unpredictable
token. A second action queued before its observation arrives therefore cannot
reuse the old token. Stale or invalid actions execute no physics and are recorded.
The token is a sequencing receipt, not an authentication credential.

The initial allowlist includes tick, speed, gear, lean, done and the observation
token. Perfect future curvature, tire friction, reward internals and ideal racing
lines are not exposed. Godot still records privileged state for verification.
Receipts vary between candidates; initial physical state remains identical. This
is not a claim that all prompt tokens or sampled trajectories are identical.

## Trainer authority and failure handling

TRL treats `reset()` and `get_reward()` as lifecycle methods and does not expose
them as tools. Reset assigns the trainer's policy step identifier and creates a
new episode. Reward comes from actual recorded progress after the actions. The
current `interactive-launch-v1` reward is only a short launch diagnostic; it must
be replaced by a versioned, adversarially tested racing objective before lap
optimization.

Transport errors and malformed simulator responses persist an infrastructure
failure. They make `get_reward()` raise, aborting the training update. This matters
because TRL catches exceptions raised by tools and normally returns them to the
model. A broken simulator must not silently become a poor driving score.

The adapter records reset observations, requested controls, before and after
telemetry, episode identity, policy step and reward. Godot independently records
every physical transition. The validation trainer saves exact completion token
IDs, completion masks, tool masks, advantages and episode identities. It audits
observation before action ordering and excludes intermediate tool observations
from the policy loss.

## Reproduce the interactive check

From the repository root:

```sh
python3 training/test_agent_harness.py
uv run --project training python training/interactive_trl.py \
  --godot /absolute/path/to/godot \
  --output artifacts/a-new-interactive-check
```

The script uses TRL `GRPOTrainer.environment_factory`, four isolated workers,
eight total rollouts and two optimization steps. Every rollout must complete two
observed control actions. The local model is Qwen3 0.6B because its native chat
template passes TRL's tool support check. The previously tested Gemma 3 270M
template does not. This validates the framework and harness, not Gemma 4 support.

## Remaining work

Camera observations and their calibration must join this tool flow with explicit
capture ticks. Add a telemetry or vision experiment specification, longer budgets,
an invalid command budget, and a racing reward covering lap validity and timing.
Harbor integration should wrap this same environment and independently verify its
recordings. Checkpoint reload and evaluation must be validated for the interactive
model. The earlier Gemma launch probe validates reload for that separate setup.
Longer control horizons and rollout throughput need measurements on target
hardware. None of these are implied by a two action smoke test.

## Verified local result

The [committed result](results/rtx2080ti-trl-interactive.json) records a passing
run on the RTX 2080 Ti with TRL 1.13.0. Four workers completed eight rollouts,
each with two sequential observed control actions, for 192 physics transitions.
Two GRPO updates produced finite nonzero adapter changes. The transcript audit
verified that the intermediate observation precedes the next action, that its
tokens are excluded from policy loss, and that each episode's tick sequence,
policy identifier and reward agree with Godot's independent recording. The run
excluded 1,219 environment tokens. Peak PyTorch allocation was 6,382,137,856 bytes,
about 5.94 GiB, not total GPU memory use. Full artifacts are retained under
`artifacts/interactive-trl-03/`.

Five independent harness tests cover read only observations, stale receipts,
control bounds, transport failure invalidation and policy authority limits.
The adapter was saved; this interactive test does not claim a reload check.
