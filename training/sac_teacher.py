"""A trained SAC actor as a NumPy teacher: the deterministic action for any observation.

Rev learns to ride by imitation with DAgger: the teacher labels every state, including
the states Rev reaches through its own mistakes, with the action it would take there.
The label is the actor's mean action (tanh of the mean head), not the exploration
samples SAC recorded while training.

Loading reads the sac_async checkpoint once with torch; acting is NumPy only, batched
or one observation at a time.

Teachers maps each circuit to its best teacher: the circuit's specialist from the
sac_specialists.py registry (runs/sac/specialists.json), else the general multi-track
policy, which laps most circuits. The registry is read once, so every label of one
dataset or collection comes from one snapshot of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "runs/sac/specialists.json"
GENERAL = ROOT / "runs/sac/sac-multitrack-1/checkpoints/step_40672729.pt"


class SacTeacher:
    def __init__(self, checkpoint: Path | str, pedal_gain: float = 1.25):
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["sac"]["actor"]
        trunk = sorted({int(k.split(".")[1]) for k in state if k.startswith("trunk.")})
        self.layers = [(state[f"trunk.{i}.weight"].float().numpy().T, state[f"trunk.{i}.bias"].float().numpy())
                       for i in trunk if f"trunk.{i}.weight" in state]
        self.mean = (state["mean.weight"].float().numpy().T, state["mean.bias"].float().numpy())
        self.pedal_gain, self.checkpoint = pedal_gain, str(checkpoint)

    def raw(self, obs) -> np.ndarray:
        """The policy action in [-1, 1]^2, as sac_env.step takes it."""
        x = np.asarray(obs, dtype=np.float64)
        for weight, bias in self.layers:
            x = np.maximum(x @ weight + bias, 0.0)
        return np.tanh(x @ self.mean[0] + self.mean[1])

    def controls(self, obs) -> np.ndarray:
        """(steer, applied pedal): the pedal after sac_env.controls' gain and clip."""
        a = self.raw(obs)
        return np.stack([a[..., 0], np.clip(self.pedal_gain * a[..., 1], -1.0, 1.0)], axis=-1)


class Teachers:
    """The best available SAC teacher for each circuit: its specialist, else the general policy."""

    def __init__(self, registry: Path | str = REGISTRY, general: Path | str = GENERAL, pedal_gain: float = 1.25):
        registry = Path(registry)
        entries = json.loads(registry.read_text()) if registry.exists() else {}
        self.specialists = {track: ROOT / entry["checkpoint"] for track, entry in entries.items()}
        self.general, self.pedal_gain = Path(general), pedal_gain
        self.loaded: dict[Path, SacTeacher] = {}

    def checkpoint(self, track: str) -> Path:
        return self.specialists.get(track, self.general)

    def describe(self, tracks) -> dict:
        """track -> the checkpoint that labels it, repo-relative, for summaries."""
        return {track: str(self.checkpoint(track).relative_to(ROOT)) for track in tracks}

    def __getitem__(self, track: str) -> SacTeacher:
        path = self.checkpoint(track)
        if path not in self.loaded:
            self.loaded[path] = SacTeacher(path, self.pedal_gain)
        return self.loaded[path]

    def controls(self, obs, tracks) -> np.ndarray:
        """(steer, applied pedal) for each observation from its own circuit's teacher."""
        obs, tracks = np.asarray(obs), np.asarray(tracks)
        out = np.empty((len(obs), 2))
        for track in np.unique(tracks):
            rows = tracks == track
            out[rows] = self[str(track)].controls(obs[rows])
        return out
