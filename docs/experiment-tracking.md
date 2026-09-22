# Experiment graphs

Full trajectory training records W&B scalar history locally by default. It does
not require an account or contact W&B until online mode or sync is explicitly
selected. Recordings and model weights stay in the existing artifact archive.

Graphs use `generation` as their horizontal axis. Evaluation is greedy and
separate from sampled rollouts. Completion rate and crash rate are per episode;
invalid call rate divides rejected calls by all attempted calls, including the
rejected call. `episode_seconds` measures every attempt. `lap_seconds` is present
**only for independently audited, successful laps**. Short crashes cannot appear
as fast laps. Training graphs include reward, progress, loss, gradient norm,
trained actions and tokens. Model revision and adapter SHA256 identify the policy.

The trainer accepts `--wandb-mode offline` (default), `online`, or `disabled`,
plus `--wandb-project` and `--wandb-entity`. Offline files survive a training
exception and can be synced later. `tracking/metrics.jsonl` provides the same
scalar events in readable form. Live collection progress is logged every 100
control decisions; aggregate outcomes are logged after collection and updates.

Import a historical campaign without running the simulator or uploading assets:

```sh
uv run --project training python training/experiment_tracking.py \
  path/to/experiment/campaign.json \
  --output artifacts/tracking/my-campaign \
  --name my-campaign
```

The importer also includes completed generation collections whose optimizer
update was interrupted. It does not invent a completed update for those attempts.
Use a fresh output directory for each import. Reimporting creates a separate run.

Once a W&B account and project are selected, authenticate interactively and sync
only the desired offline run directory printed by the importer:

```sh
uv run --project training wandb login
uv run --project training wandb sync path/to/wandb/offline-run-TIMESTAMP-ID
```

Alternatively pass `--mode online --entity YOUR_ENTITY` to a historical import,
or `--wandb-mode online --wandb-entity YOUR_ENTITY` to the trainer. Online mode
requires credentials; never put an API key in the repository or command arguments.

The SDK is pinned in `training/pyproject.toml`. See the official
[offline tracking documentation](https://docs.wandb.ai/models/track/environment-variables)
and [SDK release](https://pypi.org/project/wandb/0.30.0/).
