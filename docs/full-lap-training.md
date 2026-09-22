# Full lap training

`training/train_full_lap_grpo.py` samples independent standing start episodes from a frozen Gemma policy. Each episode ends on lap completion, crash, track violation, malformed control, or its simulation time budget. Every action retains its original stateless road telemetry prompt, generated token IDs and behavior log probabilities. No teacher controls enter collection.

TRL 1.13.0 computes Dr GRPO loss for every sampled action. Episode rewards are centered within the generation, without standard deviation scaling. Accumulated loss uses a fixed denominator of episode count times the maximum actions times the native protocol completion token limit. Prompt and padding tokens are excluded. One optimizer step follows the complete generation. The optimizer persists between generations and is saved with each adapter.

Reward v2 pays signed legal progress in meters divided by 100. An independently audited valid lap also earns a completion bonus of 2 and a time bonus between 0 and 1. Crashes and track violations cost 0.2; invalid syntax costs 1. Stalls preserve earned reward with no additional failure penalty. After a five second launch grace period, less than one meter of signed legal progress over five seconds ends an attempt. See `reward-references.md` for source comparisons and limitations.

## Local learning diagnostic

The bounded local campaign uses FunctionGemma 270M with fresh LoRA initialization, native constrained tools and telemetry. It performs 20 optimizer updates from 12 sampled attempts per update. Inference batch size 4 means four waves of three sampled attempts plus a greedy evaluation lane. Weights stay frozen across all waves, and advantages are centered across all 12 training attempts together. Each attempt allows 30 simulated seconds, subject to early termination. This measures initial driving improvement, not full lap skill.

The starting policy and checkpoints 5, 10, 15 and 20 each receive 12 separate sampled evaluation attempts. The evaluation sampling stream resets to seed 1073 at each checkpoint and restores the training RNG on exit. Training uses seed 73. Evaluation attempts never enter optimizer updates. Both use temperature 3; greedy behavior is measured separately. W&B `heldout/*` metrics compare these evaluations, including median legal progress, completion, stall and track limit rates. `rollout/*` metrics describe training attempts. All attempts retain native decision logs, simulator recordings and queued videos.

```sh
uv run --project training python -u training/local_native_validation.py \
  --godot GODOT_PATH --output artifacts/UNIQUE_LOCAL_RUN \
  --generations 20 --rollouts-per-generation 12 --batch-candidates 4 \
  --time-budget-seconds 30 --temperature 3 \
  --evaluation-interval 5 --evaluation-rollouts 12 --evaluation-seed 1073 \
  --wandb-mode online --wandb-entity skeptrune-org
```

Judge sustained changes in the separate evaluation distribution rather than the best training attempt. Twenty generations is a diagnostic budget, not a promise of learning. A second training seed is needed before claiming reproducible improvement.

Inference and backward microbatches are measured separately. Inference requires compiled static cache generation; silent eager fallback fails qualification. The campaign selects measured valid control throughput with 15 percent device memory headroom. Backward probes reserve future AdamW state memory. At most 64 simulator lanes are considered for the 16 CPU, 128 GiB host. One lane provides greedy evaluation and is excluded from training.

Each generation saves an adapter, optimizer state, loss coverage audit, all simulator recordings, decision traces and video jobs. A second adapter load verifies saved checkpoint logits. A final greedy episode evaluates the last update. Recordings must be downloaded completely outside the local video watcher root, then atomically published there for rendering with `tools/render_video_queue.py`.

Run a short integration test first:

```sh
modal run --detach training/modal_app.py --stage full-lap-smoke --run-id UNIQUE_SMOKE_ID --source-run gemma4-warmstart-rl-01
```

Then launch three generations, each allowing a complete lap up to 900 simulation seconds:

```sh
modal run --detach training/modal_app.py --stage full-lap --run-id UNIQUE_CAMPAIGN_ID --source-run gemma4-warmstart-rl-01
```

The cloud function has a twelve hour wall time limit and commits artifacts every minute. A timeout interrupts only the trainer first, allowing simulator recording closure. A short smoke test is infrastructure verification, not evidence of a completed lap or improved racing performance. Compare completion rate, audited lap times, failures and greedy evaluation across the preserved generations before judging improvement.

Completed runs can be archived and rendered automatically. Staging must be on the same filesystem as the destination, outside the watched artifacts tree:

```sh
DISPLAY=:1 python3 tools/archive_modal_run.py --run-id RUN_ID --staging-root ../thunderhill-downloads --output artifacts/modal-RUN_ID-archive --godot GODOT_PATH --ffmpeg FFMPEG_PATH --watch
```

The archive helper waits for terminal status, downloads through the Modal CLI, validates run identity, publishes the directory atomically and drains all video queues. It retains failed training attempts too.
