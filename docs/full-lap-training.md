# Full lap training

`training/train_full_lap_grpo.py` samples independent standing start episodes from a frozen Gemma policy. Each episode ends on lap completion, crash, track violation, malformed control, or its simulation time budget. Every action retains its original stateless road telemetry prompt, generated token IDs and behavior log probabilities. No teacher controls enter collection.

TRL 1.13.0 computes Dr GRPO loss for every sampled action. Episode rewards are centered within the generation, without standard deviation scaling. Accumulated loss uses a fixed denominator of episode count times the maximum actions times 32 completion tokens. Prompt and padding tokens are excluded. One optimizer step follows the complete generation. The optimizer persists between generations and is saved with each adapter.

An independently audited valid lap earns 3 plus a time bonus between 0 and 1. Incomplete attempts earn at most 1 for net legal progress. Crashes, track violations and malformed controls incur penalties. Faster valid completions therefore score higher without allowing a short failed attempt to beat a complete lap.

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

The cloud function has a six hour wall time limit and commits artifacts every minute. A timeout interrupts only the trainer first, allowing simulator recording closure. A short smoke test is infrastructure verification, not evidence of a completed lap or improved racing performance. Compare completion rate, audited lap times, failures and greedy evaluation across the preserved generations before judging improvement.
