# NVIDIA pipeline audit

Audit date: September 24, 2026. Game revision: `3151354`.

## Conclusion

We reuse NVIDIA's public AlpaGym training implementation, with a custom Godot
simulator adapter. We do not reproduce NVIDIA's complete original paper training
procedure, simulator, vehicle, observations, or reward task. The optimizer code
is genuinely upstream. Successful GPU training of this integration is unverified.

The most consequential findings are a confirmed controller coordinate problem,
unused navigation input in the pinned native model path, a frozen language and
vision backbone, and smoke configuration defaults that should not be mistaken
for a qualified long training recipe. This audit changes no learning behavior.

## Scope and source identity

The comparison is against these inspected checkouts:

| Source | Revision |
| --- | --- |
| AlpaGym | `972d160eed0e23d388497851504a3a233fec5879` |
| Alpamayo | `4cda35d22bb257f0936ac272397627b9309ca211` |
| Alpamayo recipes | `1d99fc50370c96637157455da386323a44663c9d` |
| AlpaSim | `46b401beaec27977cbb86826df8e39cec2d49e49` |
| Cosmos RL | `d2a2c57c4bd6496482bc42da19a59b4432705eda` |

NVIDIA publishes multiple related approaches. The original Alpamayo paper
includes reasoning quality, reasoning/action consistency and trajectory quality
rewards. The public AlpaGym recipe performs simulation driven trajectory expert
RL. We implement the latter. Public code parity cannot establish equivalence to
unpublished internal training runs.

Primary public references:

* [AlpaGym pinned onboarding](https://github.com/NVlabs/alpagym/blob/972d160eed0e23d388497851504a3a233fec5879/docs/ONBOARDING.md)
* [NVIDIA explanation of the public simulation RL workflow](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
* [Alpamayo paper, section 5.3](https://arxiv.org/html/2511.00088v2)

## Pipeline comparison

| Stage | Our implementation | Comparison and implication |
| --- | --- | --- |
| Starting model | Converted Alpamayo 1.5 expert checkpoint supplied by path | Same expected architecture and conversion machinery. Actual checkpoint loading in the rewritten GPU pipeline remains untested. |
| Trainable parameters | Native ExpertModelCosmos wrapper | Language and vision are frozen. Trajectory expert RL, not language reasoning RL. |
| Scene source | Single standing start at Thunderhill | Custom scene instead of recorded NVIDIA road scenes. No scenario diversity or separate evaluation suite configured. |
| Simulator | Explicitly stepped Godot motorcycle physics | Replaces AlpaSim, rendering, vehicle dynamics and controller. Physics parameters include engineering estimates and are not validated Ducati dynamics. |
| Camera | One rider camera | Official selected preset uses four cameras. Bike mounting, lean, field of view and synthetic appearance differ. |
| Sensor history | Measured 1.5 second stationary warmup | Native 16 pose history and four image context frames; replaces recorded motion warmup. No teacher driving. |
| Timing | Images and poses every 0.1 seconds, plans every 0.2 seconds | Matches the selected public smoke example's control interval. Simulation pauses for inference. |
| Protocol | Real AlpaSim RuntimeService and EgodriverService | Native protocol and policy preprocessing, with Godot implementing the environment side. |
| Navigation | Twenty centerline points over 80 meters sent in rig coordinates | Route protocol is populated, but the pinned expert path does not consume it. |
| Actions | Native stochastic future trajectories, 64 points at 0.1 seconds | NVIDIA action representation. These are not sampled language tool calls. |
| Execution | Custom pure pursuit steering and PI speed controller | Different from NVIDIA MPC and car dynamics; confirmed roll/frame problem below. |
| Reward | Native reward dispatcher with custom game metrics | Progress units fixed, but task semantics remain different. No reference trajectory penalty. |
| Episode endings | Lap, fall, track invalidation, stall or time limit | Custom early stops; failed driving remains training experience. One lap ends a rollout. |
| Rollout grouping | Native backend, with recording identity wrapper | Full sibling group retained. Wrapper delegates generation to upstream. |
| Replay | Native action samples, SDE trace, old likelihood and exact model inputs | Actual upstream replay machinery; not fitting teacher controls. |
| Advantages | Native Cosmos GRPO | Relative rewards within siblings; identical rewards produce zero advantage. |
| Objective | Native clipped policy objective with padding masks | No custom likelihood or optimizer implementation. Actual GPU gradients unverified. |
| Weight exchange | Native NCCL | Selected architecture uses separate training and rollout GPUs. Synchronization not yet exercised in the rewritten runtime. |
| Checkpoints | Native model, optimizer and scheduler checkpoint code | Final step exports safetensors even though smoke preset disables ordinary intermediate exports. Reload/evaluation untested here. |
| Tracking | Native Cosmos console and W&B logging | Generic reward/loss logging configured. Custom lap outcomes and component dashboards are not wired as a verified hosted evaluation suite. |
| Recordings | Simulator states, images, predicted trajectories, executed controls and video jobs | Strong local provenance. MP4 rendering is a separate command, not automatically launched with training. |

## Confirmed findings

### 1. Controller frame mismatch must be corrected before interpreting cornering RL

`training/alpagym_bridge.py:56` preserves measured roll and pitch. Its future
trajectory conversion at line 89 returns points in that rig frame.
`training/driving_trajectory.py:135` then drops Z and calculates planar curvature.
The controller assumes a level plane while the input frame can be banked.

A direct reproduction using the existing controller represented the same flat
world curve in an upright frame and a frame rolled 45 degrees. At 10 m/s,
computed curvature changed from 0.0199025 to 0.0141076 per meter, approximately
29 percent lower. Steering changed from about negative 0.22746 to negative
0.16231. No model sampling was involved.

The model must continue receiving truthful measured motion. Correct the
controller's frame projection separately and verify curved, banked and elevated
trajectory tracking. Straight speed hold and stopping tests do not cover this.
NVIDIA's controller uses MPC and full vehicle state; ours uses speed and lean
with approximate motion between replans. That difference can punish a correct
model plan for an execution error.

### 2. Route points do not condition the selected native model

The bridge sends route points, and upstream `inference_model.py:483` carries
`route_xy` in an intermediate dictionary. However:

* `alpagym_alpamayo_r1/tokenize_online.py:77` selects image, history, prompt and future components without a route component.
* The sample assembled at line 89 contains images and camera IDs.
* The recipes expert implementation consumes tokenized inputs and ego history, with no route consumption in that model subtree.
* `alpagym_alpamayo_r1/inference_model.py:342` also omits route from training inputs.

Therefore the previous explanation that centerline navigation might directly
bias the policy was incorrect for this pinned path. The model currently relies
on visual and motion history conditioning. Centerline geometry still measures
simulator progress. No optimized racing line or proximity reward exists.

Adding navigation only to inference would be insufficient: training replay must
reconstruct the same conditioning. A future route integration needs a paired
input sensitivity test and inference/replay agreement.

### 3. The language model is not being fine tuned by this recipe

`alpamayo-recipes/recipes/alpamayo1_x_rl/models/expert_model/cosmos_wrapper.py:43`
sets `cotrain_vlm=False` and freezes language and visual parameters. AlpaGym's
`bundle.py:72` explicitly rejects reasoning token replay. Its
`inference_model.py:123` skips autoregressive VLM generation on the selected path.

This is genuine RL on the trajectory expert, using the model's own gameplay.
It does not train verbal decisions or native text tool calling. The cockpit
control receipts describe the controller's executed actions, not generated
reasoning. Helper names such as `sft_process_sample` in upstream tokenization
are preprocessing reuse, not evidence that this workflow performs supervised
imitation. No demonstration loss is introduced by our adapter.

### 4. Reward units match, but reward behavior does not equal NVIDIA's task

Current scalar reward is:

```
clip(legal forward meters / full circuit meters, 0, 1)
  - 10 * obstacle collision
  - 5 * any offroad event
  - 10 * fall without obstacle collision
```

`training/alpagym_metrics.py` supplies these metrics to the unchanged native
reward dispatcher. Every executed physics transition contributes failure flags.
Warmup progress is excluded. Old prepared configurations are rejected.

NVIDIA's `GroundTruthScorer` projects the vehicle centroid onto the complete
recorded scene trajectory, normalized to its length. We accumulate signed legal
circuit progress. NVIDIA uses vehicle footprint geometry for relevant safety
checks; our on_track flag checks the bike position against track width. NVIDIA's
progress_safety preset additionally penalizes timestamp aligned deviation from
recorded expert motion. We omit that term and add the motorcycle fall term.

The fixed full lap denominator is stable, but short attempts earn a small
fraction of the maximum progress reward while failures retain their full cost.
That changes the tradeoff relative to NVIDIA's short recorded scenes. GRPO
normalization prevents us from concluding that a small absolute progress value
alone produces a small learning signal. The more significant concern is reward
ordering between cautious stopping and risky movement.

Speed is only encouraged indirectly by distance within the time budget. A bike
that stalls can keep a small positive progress score; a bike that travels farther
then falls receives a large negative score. Identical stalls yield no relative
advantage. Once clean laps are completed, the formula contains no explicit lap
time preference; progress differences from stopping/gate discretization are not
an intentional speed objective. We therefore cannot call this an optimal racing
reward. The historical speed/completion terms in `lap_episode.py` are not used by
this AlpaGym path; only its StallMonitor is imported.

### 5. Episode design is a short test, not the requested full lap curriculum

`training/nvidia_alpagym.py:53` defaults to one training step, two siblings,
30 simulated seconds and concurrency one. `godot/scripts/main.gd:811` ends an
agent episode after a completed lap. Longer episodes are configurable, but
multiple continuous laps per rollout are not currently implemented.

The bridge ends on invalid lap status and a stall of less than one meter net
progress over five seconds after a five second grace period. Crash stop reasons
can be labeled `track_limits` because lap invalidation is checked before the
terminal reason (`alpagym_bridge.py:587`). Collision/fall metrics still preserve
the event, but a chart grouped only by stop_reason would misclassify it.

### 6. Long training would inherit smoke hyperparameters

The selected official preset has learning rate 1e-4, one warmup step, decision
minibatch size one, one optimization pass and zero KL coefficient. Its inference
maximum batch size is one and rollout prefetch is disabled. The generic upstream
configuration uses different learning rate and warmup defaults. Neither is
established as a good motorcycle training recipe by the current evidence.

NVIDIA's trainer flattens valid decision rows and copies each episode advantage
to every row. With two complete 30 second attempts at five decisions per second,
there can be 300 decision minibatches per training step. These are not 300
independent rollouts. Video generation is the actual policy weight version,
not a count of individual optimizer minibatches.

Short failures are padded to the configured expected decision count. Padding has
zero advantage and is masked from the policy objective, but still consumes
forward compute. Simply extending the horizon substantially can spend time on
padding if most attempts stop early. Increasing simulator concurrency alone
also does not increase the inherited inference batch limit.

### 7. Runtime readiness is not established by current local tests

The intended official environment at
`../thunderhill-references/alpagym/.venv/bin/python` is absent on this machine.
`tools/setup_alpagym.py` defines the frozen dependency installation, while the
launcher uses `uv run --no-sync`. CPU tests currently use the retained ignored
local environment. That is sufficient for those tests, not proof of the official
CUDA stack being installed or compatible.

The launcher checks two visible CUDA devices, but that alone does not verify
memory capacity, checkpoint conversion, NCCL communication, native dependencies,
renderer availability or W&B permissions. Launch completion deliberately does
not claim optimizer verification. No current GPU update or learning improvement
was established by this audit.

### 8. Recording and evaluation still have gaps

The bridge retains image receipts, exact JPEGs sent to the driver, predicted
trajectories, control receipts and state recordings. Its wrapper snapshots the
actual Cosmos policy version and refuses prefetch that could corrupt identity.
Driving failures remain valid experiences; infrastructure failures are rejected.

The display model name is hardcoded at `alpagym_bridge.py:284`; it is not derived
from the supplied checkpoint identity. Rollout numbers count attempts across the
runtime rather than restarting per generation. These details need documentation
or adjustment before comparing multiple base models.

MP4 jobs are durable, but the training launcher does not run the renderer.
The existing interrupted recording recovery command expects older
`status.json`/`launch.json` manifests, not this launcher's
`run_status.json`/`launch_manifest.json`. Graceful failures normally publish jobs;
hard process interruption is not covered by an equivalent verified recovery path.

W&B is configured for generic upstream reward and optimizer signals. The scalar
reward callback returns only the total; custom game component metrics remain in
summaries and are not explicitly connected to Cosmos report_metrics. There is no
verified fixed seed baseline versus checkpoint evaluation with lap completion,
lap time, falls, offroad rate and videos joined in a single report. Old runs from
the removed trainer are not evidence of learning in the rewritten pipeline.

## Required validation sequence

1. Correct controller frame handling and qualify geometric tracking through corners, changing lean, slopes, acceleration and braking. Test references are controller fixtures, never training demonstrations.
2. Decide explicitly whether visual driving without route conditioning is acceptable. If navigation is required, connect it consistently through inference and replay before training.
3. Install the pinned GPU environment and load the real converted checkpoint. Inspect exactly what the real policy sees and compare native model plans with actual motion.
4. Collect a small genuine sibling group. Verify failed driving is retained, infrastructure failure is excluded, rewards differ where expected, and replay likelihood before any update matches collection likelihood within a stated numeric tolerance.
5. Perform a bounded real update. Verify finite nonzero gradients when advantages differ, changed expert weights, unchanged frozen backbone weights, checkpoint creation and synchronized next generation.
6. Reload that checkpoint and evaluate against the starting checkpoint on identical scenarios and seeds. Compare gameplay metrics and recordings. Only then increase horizon, generations and rollout concurrency with measured throughput.
7. Before optimizing racing speed, choose an explicit time objective and test its ordering on safe laps, stalls, crashes and shortcuts. Plot any proposed racing reference before using it, as requested.

No new racing line, supervised training, paid GPU allocation or reward change was
performed for this audit.

## Verification performed for this audit

Before the requested unit test cleanup, 27 local checks completed with two
Godot checks skipped in that invocation. A separate invocation then passed both
the real Godot recording check and the official NVIDIA driver/preprocessing
check. The latter uses synthetic inference, not model weights. The controller
frame reproduction was also executed directly. These results validate the
stated protocol boundary, not GPU optimization or driving improvement.

The user subsequently requested removal of standalone unit tests. That cleanup
retains actual game integration checks; the counts above are a historical audit
record and not instructions to restore the deleted suite.
