# Modal validation

The first cloud stage runs the larger Gemma diagnostic on one H100. It does not start a long training campaign or reuse the small model's adapter. The diagnostic must establish that the larger model can emit bike controls and perform an actual simulator rewarded update before scaling up.

From the repository root:

```bash
modal run --detach training/modal_app.py --run-id gemma4-diagnostic-01
```

Every launch requires a fresh run ID. Existing output directories are never overwritten. `--stage diagnostic` is the only supported stage. The function allows one container, no automatic retries, and at most 30 minutes. Its subprocess gets 28 minutes, leaving time to flush recordings and persist failure information.

The image installs Python 3.12 and the exact dependencies in `training/uv.lock` with `uv sync --frozen`. It includes Godot 4.7.2, the game source, track data, and assets, and imports the project before launch. By default it reads the same Godot binary used locally from `~/.local/share/thunderhill-tools/godot-4.7.2/Godot_v4.7.2-stable_linux.x86_64`. Set `THUNDERHILL_GODOT` to that binary's location if different on the launch machine.

Artifacts live on the `thunderhill-runs-v2` Modal Volume, under the run ID. The `launch.json`, `run.log`, and `status.json` describe the cloud invocation; `experiment/` contains the diagnostic's complete outputs. Model downloads are cached separately on `thunderhill-huggingface-v2`. Both use Volume v2 because publishing immutable video jobs requires hard links. Volumes commit in the background and explicitly at the end of the function.

Inspect the run using the app URL printed by the CLI or `modal app logs` with its app ID. Download the entire run after completion, including failures:

```bash
modal volume get thunderhill-runs-v2 gemma4-diagnostic-01 artifacts/modal-gemma4-diagnostic-01
```

Video jobs use relative recording paths, so they remain valid after downloading. Render every queued episode locally using the existing archive command, supplying your Godot and FFmpeg executable paths:

```bash
python3 tools/render_video_queue.py \
  artifacts/modal-gemma4-diagnostic-01 \
  --godot /path/to/godot \
  --ffmpeg /path/to/ffmpeg
```

An archived rollout is not considered to have video coverage until its render receipt appears in the montage index. Downloaded raw episodes without queued jobs, failed renders, and interrupted episodes remain visible as archive failures. The cloud run does not claim to have produced MP4 files before local rendering succeeds.

See the official [Modal image guide](https://modal.com/docs/guide/images), [Volume guide](https://modal.com/docs/guide/volumes), and [function guide](https://modal.com/docs/guide/functions).
