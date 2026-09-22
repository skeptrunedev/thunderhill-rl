# Local GPU validation

Use Hugging Face TRL as the primary training framework, with PEFT adapters.
The interactive [bike harness](HARNESS.md) now uses `GRPOTrainer.environment_factory`.
The launch probe here deliberately uses the simpler reward callback to first
verify the chain from model sampling to actual Godot outcomes to GPU updates.
That original launch probe does not implement repeated observations and decisions
within an episode. The separate interactive harness does. Harbor integration,
camera training and Gemma 4 E4B training remain unverified.

Run from the repository root:

```sh
uv sync --project training --locked
uv run --project training python training/smoke_grpo.py \
  --godot /absolute/path/to/godot \
  --output artifacts/my-new-training-check
```

The output directory must be new. It retains the adapter, tokenizer, trainer logs,
model token outputs, episode identifiers and real Godot episode recordings.
Dependencies are pinned in `uv.lock`. Model files are downloaded from the pinned
`unsloth/gemma-3-270m-it` revision. They are not included in this repository;
their Gemma terms remain separate from this repository's code license.

The test uses a neutral A/B exploration prompt. A coasts and B applies throttle
0.6 for one simulated second, starting from the same stationary reset. Valid
actions receive the sum of measured legal progress from actual game transitions.
Invalid output receives an explicit minus 0.1 syntax penalty and executes no
physics. This is the diagnostic `launch-probe-v1` reward, not a racing reward.
The full vocabulary remains available to generation; actions are not forced.
Single token completions are intentional; their length cutoff remains eligible
for training rather than being masked as an incomplete answer.

Three GRPO steps sample eight candidates each. The test requires both valid
actions to produce distinct rewards within a candidate group, finite nonzero LoRA
changes, changed policy logits, saved and reloaded logit agreement, and persisted
recordings for every episode. Only LoRA parameters may be trainable. FP32 and
Transformers generation keep this small compatibility test straightforward on
Turing; this configuration is not the proposed high throughput E4B deployment.
No model or training data is uploaded to an external tracking service.

## Framework selection

[TRL](https://github.com/huggingface/trl) has an Apache 2.0 license and about
19,000 GitHub stars at the September 21, 2026 check. Its
[agent interface](https://huggingface.co/docs/trl/grpo_trainer#agent-training)
supports separate environment instances, action tools, observations and episode
rewards. The current docs list Gemma 4 E2B among tested models; exact E4B
compatibility still requires a run. The
[Harbor adapter](https://huggingface.co/docs/trl/main/en/harbor)
provides a path to the project's intended harness.

Read these concrete references at upstream commit
`ccdaa0065cee4174d2b4eb60f1856c51e51232ed`:

* [Gemma CARLA](https://github.com/huggingface/trl/blob/ccdaa0065cee4174d2b4eb60f1856c51e51232ed/examples/grpo_carla/carla_vlm_gemma.py)
  for repeated driving actions, camera observations and environment rewards.
* [Catch](https://github.com/huggingface/trl/blob/ccdaa0065cee4174d2b4eb60f1856c51e51232ed/examples/grpo_catch/grpo_catch.py)
  for grouped interactive rollouts and colocated vLLM generation.

[Unsloth](https://github.com/unslothai/unsloth) remains a candidate for reducing
E4B memory use with TRL. Its [Gemma guide](https://unsloth.ai/docs/models/gemma-4/train)
distinguishes quantized and ordinary LoRA requirements; advertised minimums are
not evidence that our camera and rollout workload fits. [verl](https://github.com/verl-project/verl)
has a real interactive agent loop, but its distributed infrastructure adds
complexity that is unnecessary for this first local test.

The local GPU is an RTX 2080 Ti with 11 GB, compute capability 7.5. The official
[E4B checkpoint](https://huggingface.co/google/gemma-4-E4B-it) is publicly
accessible as of this check. E4B denotes effective computation, not total resident
parameters. Roughly eight billion total parameters mean ordinary two byte weights
alone exceed this GPU's capacity. Quantized E4B training, longer trajectories,
vision observations and target RTX PRO throughput remain separate tests. A small
Gemma success establishes working training machinery, not racing competence.

## Verified local result

The September 21 run passed, using the pinned dependencies above. The compact
[measured result](results/rtx2080ti-gemma270m.json) is committed; complete artifacts
are in `artifacts/training-smoke-04/`.

* Three GRPO steps sampled 24 completions, including 23 valid actions.
* Four coast actions produced about minus 0.017 metres of signed progress;
  nineteen throttle actions produced about 2.896 metres. One invalid C output
  received the explicit syntax penalty. Every valid action executed 120 ticks.
* All 2,760 transitions and all episode identities were found in isolated Godot
  recordings. Reward variation included both valid actions within a group.
* The 368,640 trainable adapter parameters changed, with maximum absolute change
  0.000193. Policy logits changed and matched after checkpoint reload.
* At sampling temperature 1.5, the prompt's throttle token probability increased
  from 0.5465 to 0.8804. This measures the trained launch probe, not held out
  generalization, lap completion or improved racing skill.
* Peak PyTorch allocated memory was 2,198,125,056 bytes (about 2.05 GiB), including
  the reload check. This is not total system GPU use or an E4B memory estimate.

Earlier runs correctly failed the nonzero update gate when every candidate chose
the same action. Another run found a relative user data path error after a valid
update; the final harness resolves its artifact directory before starting Godot
and verifies the episode records explicitly. Those runs are not counted as passes.

## Road telemetry lap policy

Road telemetry is the standard observation mode for the lap policy. It includes
speed, lean, lateral position, angle and distance to a lookahead centerline point,
and upcoming curvature. Geometry is simulator supplied and explicitly privileged.
The model generates steering, throttle and both brake commands; no teacher
controller supplies or substitutes actions during evaluation.

`lap_policy.py` builds causally paired supervised examples from an audited driver
recording. Each target uses the observation before the action. A contiguous
holdout and excluded boundary rows keep adjacent augmented examples out of the
holdout. This is still one recorded lap, not an independent generalization test.
`train_lap_sft.py` uses TRL SFTTrainer to warm start the small Gemma adapter. It
checks prompt token boundaries and rejects examples that would be truncated.
This stage is imitation learning, not a policy gradient update. Recovery augmentation
adds balanced low speed, cruising and braking examples with heading and lateral
perturbations; the source lap alone is dominated by steady cruising. An existing
adapter can continue supervised training with `--adapter /path/to/adapter`.

```sh
python3 training/lap_policy.py \
  --driver /absolute/driver.jsonl --episode /absolute/episode.jsonl \
  --output artifacts/new-lap-dataset --augment 2 --max-speed 8 --recovery-grid
uv run --project training python training/train_lap_sft.py \
  --dataset artifacts/new-lap-dataset --output artifacts/new-lap-sft \
  --steps 600 --batch 2
uv run --project training python training/evaluate_lap.py \
  --adapter artifacts/new-lap-sft/adapter --godot /absolute/path/to/godot \
  --output artifacts/new-lap-evaluation
```

The evaluator reloads the saved adapter into a new process and obtains fresh
telemetry after each control interval. It saves exact generated tokens, prompts,
parsed controls, adapter identity, harness events and authoritative Godot state
recordings. Malformed commands stop evaluation rather than triggering a fallback.
A completed lap must pass ordered gates, track validity and recording checks.
Training loss alone is not evidence of a successful lap.


For a long local CUDA evaluation, add `--compile` to `evaluate_lap.py`. The
compiled mode pads prompts on the left to 256 tokens and uses a 288 token static
cache. Full graph compilation uses `dynamic=True` because Gemma's sliding cache
increments a Python position counter during generation. Making that counter
static forces recompilation at each token. Compilation failures are not hidden.

The [measured inference check](results/rtx2080ti-gemma-inference.json) compared 24
generations over 12 held out prompts. Compiled outputs matched eager tokens
exactly, with average control generation falling from 310 to 128 milliseconds.
These timings include another lap evaluator sharing the GPU. A separate profiler
check verified 12 CUDA graph launches during one 13 token control generation.
The evaluator also checks adapter merge logit equivalence on its initial state.

```sh
TORCH_LOGS=perf_hints uv run --project training python training/benchmark_lap_inference.py \
  --adapter /absolute/adapter --prompts /absolute/eval.jsonl \
  --output artifacts/new-inference-check --samples 12 --repeats 2 --profile
```

`--device cpu` supports independent checks while GPU training is running. It is
not the recommended throughput path. All optimizer checkpoints are retained by
new training runs so their generated driving episodes can be compared later.


## Initial moving segment RL result

`train_lap_grpo.py` performs actual TRL GRPO updates from Godot outcomes, starting
from a supervised adapter. A frozen model generated prefix puts all candidates
into the same moving state. Four sampled first controls are followed by greedy
controls from the current model. Only the sampled first command receives a
policy gradient; neither the prefix nor continuation tokens enter the loss.
Progress after the prefix supplies reward, with separately recorded syntax and
invalid lap penalties. This is a limited Monte Carlo first decision experiment,
not full trajectory optimization.

The [measured run](results/rtx2080ti-gemma-road-grpo.json) passed three updates and
12 sampled rollouts, independent recording and loss mask audits, finite adapter
changes and checkpoint reload. Its tiny reward change on an easy straight is not
meaningful evidence of better driving. The supervised lap baseline reached about
40 percent before leaving the track in a tight corner. The subsequent corner
training result below supersedes that failed baseline.


## First verified learned lap

[Generation 10 completed a full lap](results/rtx2080ti-gemma-first-lap.json) in
600.925 simulated seconds after ten corner GRPO updates (forty sampled rollouts).
The reloaded checkpoint generated all 6,010 controls from road telemetry, with
no teacher or fallback controller. Independent audit verified all 32 gates in
order, 72,111 contiguous physics transitions, and zero off track ticks.

This is a slow baseline on the training track, not evidence of general racing
ability or improved lap time. Subsequent speed training must beat this complete
legal lap under the same standing start and simulator conditions. Checkpoint
`5618108d3ad901be680fb7e745dc3a753c4a651821a07698daa4c2768f0bd8b6`
is retained unchanged for comparison.

`train_lap_curriculum.py` can alternate compiled full lap evaluation with
training on audited failures. Every trained checkpoint is evaluated, including
the last cycle. `--start-generation` preserves generation numbering across
training calls. Successful completion ends this safety curriculum; speed
experiments then use fixed duration sections and compare complete lap times.
