"""Single GPU Alpamayo action expert RL from the policy's own game attempts.

Uses NVIDIA's stochastic flow matching sampler and differentiable replay density.
The VLM remains frozen. No demonstration labels or supervised loss are involved.
See AlpaGym's ExpertModelRL and release checkpoint conversion for the reference.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

MODEL_ID = "nvidia/Alpamayo-1.5-10B"
MODEL_REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
PROCESSOR_ID = "nvidia/Cosmos-Reason2-8B"
PROCESSOR_REVISION = "a9fae2cf89dc64db96b12860417f0eb403013bb9"
SOURCE_REVISIONS = {
    "alpamayo": "4cda35d22bb257f0936ac272397627b9309ca211",
    "alpamayo-recipes": "1d99fc50370c96637157455da386323a44663c9d",
    "alpagym": "972d160eed0e23d388497851504a3a233fec5879",
}


def _tree_to(value: Any, device: str):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: _tree_to(item, device) for key, item in value.items()}
    return value


def configure_sources(source_root: Path) -> None:
    """Verify checkout identity before loading any upstream Python source."""
    import subprocess
    for name, expected in SOURCE_REVISIONS.items():
        actual = subprocess.check_output(
            ["git", "-C", str(source_root / name), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            raise ValueError(f"Unpinned upstream {name}: {actual}, expected {expected}")
    for rel in (
        "alpamayo/src", "alpamayo-recipes/recipes", "alpamayo-recipes/src",
        "alpagym/packages/policies/alpamayo_r1/src",
    ):
        sys.path.insert(0, str(source_root / rel))


CONFIG_DERIVED_BUFFERS = frozenset({
    "action_in_proj.sinus.0.freqs", "action_in_proj.sinus.1.freqs",
    "action_in_proj.timestep_fourier_encoder.freqs", "action_space.accel_mean",
    "action_space.accel_std", "action_space.curvature_mean", "action_space.curvature_std",
})


def restore_release_buffers(model, load_info: dict, converted_config: dict) -> dict:
    """Reconstruct the seven nonlearned release buffers, never missing weights.

    Alpamayo 1.5 declares these persistent=False; the pinned AlpaGym-compatible
    R1 classes declare them persistent=True. Release safetensors therefore omit
    them. HF meta loading may leave missing buffers uninitialized, so accepting
    the missing-key warning alone is insufficient. Recreate from the original
    constructor inputs and copy real values before any inference.
    """
    import torch
    from hydra.utils import instantiate

    missing = set(load_info.get("missing_keys", ()))
    problems = {key: load_info.get(key) for key in (
        "unexpected_keys", "mismatched_keys", "error_msgs"
    ) if load_info.get(key)}
    unknown = missing - CONFIG_DERIVED_BUFFERS
    if unknown:
        problems["missing_keys"] = sorted(unknown)
    if problems:
        raise RuntimeError(f"Release checkpoint did not load exactly: {problems}")
    if not missing:
        return {"restored": {}, "reason": "no_missing_buffers"}
    if converted_config["action_in_proj_cfg"]["_target_"] != (
        "alpagym_alpamayo_r1.submodules.action_in_proj.PerWaypointActionInProjV2"
    ) or converted_config["action_space_cfg"]["_target_"] != (
        "alpamayo_r1.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace"
    ):
        raise ValueError("Buffer reconstruction requires the verified release module classes")
    # The projection constructor initializes temporary learned layers too. Save
    # CPU RNG so reconstructing constants does not alter subsequent LoRA seeds.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        projection = instantiate(
            converted_config["action_in_proj_cfg"],
            in_dims=model.action_in_proj.in_dims, out_dim=model.action_in_proj.out_dim,
        )
        action_space = instantiate(converted_config["action_space_cfg"])
    reference = {"action_in_proj": projection, "action_space": action_space}
    replacements = []
    for name in sorted(missing):
        root, relative = name.split(".", 1)
        expected = reference[root].get_buffer(relative).detach()
        module_name, buffer_name = name.rsplit(".", 1)
        module = model.get_submodule(module_name)
        if buffer_name not in module._buffers or buffer_name in module._parameters:
            raise RuntimeError(f"Expected registered nonlearned buffer: {name}")
        old = module._buffers[buffer_name]
        if old is None or old.shape != expected.shape or old.requires_grad:
            raise RuntimeError(f"Unexpected buffer contract for {name}")
        # Use a loaded model parameter for device placement even when HF left
        # the omitted buffer on meta; preserve the module's intended dtype.
        target = expected.to(device=next(model.parameters()).device, dtype=old.dtype)
        if target.is_meta or not torch.isfinite(target).all():
            raise RuntimeError(f"Invalid reconstructed buffer: {name}")
        replacements.append((name, module, buffer_name, target))
    restored = {}
    with torch.no_grad():
        for name, module, buffer_name, target in replacements:
            module.register_buffer(buffer_name, target, persistent=True)
            restored[name] = {"shape": list(target.shape), "dtype": str(target.dtype),
                              "values": target.float().cpu().tolist()}
    return {
        "restored": restored,
        "reason": "release_nonpersistent_buffers_reconstructed_from_converted_config",
        "release_source_revision": "36aeb4c5938cbc2eb2aed33b22434773da4ab639",
        "recipe_source_revision": SOURCE_REVISIONS["alpamayo"],
        "config_sha256": hashlib.sha256(json.dumps(
            converted_config, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }


class AlpamayoPolicy:
    def __init__(
        self, source_root: Path | None = None, *, checkpoint: Path | None = None,
        processor_path: Path | None = None, device: str = "cuda", rank: int = 8,
        diffusion_steps: int = 10, noise_level: float = 0.7,
        adapter: Path | None = None,
    ) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from peft import LoraConfig, PeftModel, get_peft_model

        source_root = Path(source_root or os.environ.get("ALPAMAYO_SOURCE_ROOT", "/opt/alpamayo-sources"))
        configure_sources(source_root)
        from alpamayo1_x_rl.models.expert_model.config import ExpertModelConfig
        from alpamayo1_x_rl.models.expert_model.model import ExpertModelRL

        self.device = device
        self.diffusion_steps = diffusion_steps
        self.noise_level = noise_level
        if diffusion_steps < 2 or noise_level <= 0:
            raise ValueError("Stochastic RL requires positive noise and at least two diffusion steps")
        self.model_path = str(checkpoint or snapshot_download(MODEL_ID, revision=MODEL_REVISION))
        processor_path = str(processor_path or snapshot_download(
            PROCESSOR_ID, revision=PROCESSOR_REVISION,
            allow_patterns=["*.json", "*.txt", "*.jinja"],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt"],
        ))
        converter_path = Path(source_root) / (
            "alpagym/packages/policies/alpamayo_r1/scripts/"
            "convert_release_to_alpagym_checkpoint.py"
        )
        spec = importlib.util.spec_from_file_location("alpagym_checkpoint_conversion", converter_path)
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
        release_config = json.loads((Path(self.model_path) / "config.json").read_text())
        converted = converter.build_expert_config(release_config, processor_path)
        converted["attn_implementation"] = "sdpa"
        self.converted_config = converted
        config = ExpertModelConfig(**converted)
        # Official ExpertModel rejects FA2: its noncausal 4D expert attention mask
        # requires SDPA. CUDA 12.8 PyTorch wheels support Blackwell fused SDPA.
        self.model, load_info = ExpertModelRL.from_pretrained(
            self.model_path, config=config, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="sdpa", output_loading_info=True,
        )
        self.buffer_restoration = restore_release_buffers(self.model, load_info, converted)
        self.model.requires_grad_(False)
        if adapter:
            metadata = json.loads((adapter / "policy_manifest.json").read_text())
            if metadata["model_revision"] != MODEL_REVISION:
                raise ValueError("Adapter base revision does not match")
            self.model.expert = PeftModel.from_pretrained(
                self.model.expert, adapter / "expert_adapter", is_trainable=True,
            )
        else:
            self.model.expert = get_peft_model(self.model.expert, LoraConfig(
                r=rank, lora_alpha=rank * 2, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ))
        # Eval mode suppresses dropout but does not suppress autograd; the same
        # stochastic policy density must be used in sampling and RL replay.
        self.model.eval()
        self.trainable = [param for param in self.model.parameters() if param.requires_grad]
        if not self.trainable:
            raise RuntimeError("No action expert adapter parameters are trainable")
        self.optimizer = torch.optim.AdamW(self.trainable, lr=1e-5, weight_decay=0.0)

    def _prepare(self, camera_frames, ego_history_xyz, ego_history_rot, instruction: str):
        import torch
        from alpagym_alpamayo_r1.data.chat_template.conversation import build_conversation

        frames = torch.as_tensor(camera_frames)
        if frames.ndim == 4 and frames.shape[0] == 4 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2).unsqueeze(0)
        if frames.ndim != 5 or frames.shape[:3] != (1, 4, 3):
            raise ValueError("camera_frames must have shape [1 camera, 4 frames, 3, H, W]")
        if frames.dtype != torch.uint8:
            raise ValueError("camera_frames must contain uint8 RGB pixels")
        sample = {"image_frames": frames.transpose(0, 1), "camera_indices": torch.ones(4, dtype=torch.long),
                  "nav_text": [instruction]}
        config = self.model.config
        messages = build_conversation(
            data=sample, num_tokens_per_history_traj=config.tokens_per_history_traj,
            num_tokens_per_future_traj=config.tokens_per_future_traj,
            components_order=["image", "traj_history", "nav_instruction", "prompt", "traj_future"],
            components_prompt=["traj_future"], generation_mode=True,
            include_camera_ids=config.include_camera_ids,
            frame_label="frame_num" if config.include_frame_nums else "none",
        )
        processor = self.model.processor
        # Matches the official packer's generation path. No labels are created.
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            add_vision_id=False, continue_final_message=True,
        )
        tokens = dict(processor(
            text=text, images=frames.flatten(0, 1).float() / 255.0,
            videos=None, padding=False, return_tensors="pt", do_rescale=False,
        ))
        xyz = torch.as_tensor(ego_history_xyz, dtype=torch.float32)
        rot = torch.as_tensor(ego_history_rot, dtype=torch.float32)
        if xyz.shape != (16, 3) or rot.shape != (16, 3, 3):
            raise ValueError("History requires 16 local ego positions and rotation matrices at 10Hz")
        if not torch.isfinite(xyz).all() or not torch.isfinite(rot).all():
            raise ValueError("Nonfinite ego history")
        return _tree_to({"tokenized_data": tokens, "ego_history_xyz": xyz[None, None],
                         "ego_history_rot": rot[None, None]}, self.device)

    def sample(
        self, camera_frames, ego_history_xyz, ego_history_rot, *, seed: int,
        instruction: str = "Race around Thunderhill East as quickly as possible while staying on the track.",
    ) -> dict:
        import torch
        data = self._prepare(camera_frames, ego_history_xyz, ego_history_rot, instruction)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            xyz, rot, old_logprob, trace = self.model._sample_trajectories_from_data_without_vlm_rollout(
                data, num_traj_samples=1, num_traj_sets=1,
                diffusion_kwargs={"int_method": "sde", "return_info": True,
                                  "inference_step": self.diffusion_steps,
                                  "noise_level": self.noise_level, "generator": generator},
            )
        if not torch.isfinite(xyz).all() or not torch.isfinite(old_logprob).all():
            raise RuntimeError("Nonfinite sampled trajectory or behavior density")
        replay = _tree_to({"data": data, "samples_list": trace["samples_list"],
                           "timesteps": trace["timesteps"].reshape(1, -1),
                           "noise_level": self.noise_level,
                           "old_logprob": old_logprob.reshape(())}, "cpu")
        return {"xyz": xyz[0, 0, 0].float().cpu().numpy(),
                "rot": rot[0, 0, 0].float().cpu().numpy(), "replay": replay,
                "old_logprob": float(old_logprob.item())}

    def logprob(self, replay: dict):
        trace = _tree_to(replay, self.device)
        logprob, _ = self.model.cfm_logprob_sde(
            trace["data"], trace["samples_list"], trace["timesteps"],
            noise_level=trace["noise_level"], teacher=None,
        )
        return logprob.reshape(())

    def adapter_hash(self) -> str:
        digest = hashlib.sha256()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                digest.update(name.encode())
                digest.update(param.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    def fingerprint(self) -> str:
        return self.adapter_hash()

    def reload(self, output: Path) -> str:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        manifest = json.loads((output / "policy_manifest.json").read_text())
        if manifest["model_revision"] != MODEL_REVISION:
            raise ValueError("Adapter base revision does not match")
        weights = load_file(str(output / "expert_adapter" / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model.expert, weights)
        digest = self.adapter_hash()
        if digest != manifest["adapter_sha256"]:
            raise RuntimeError("Reloaded adapter hash does not match saved checkpoint")
        return digest

    def update(self, episodes: list[dict], *, learning_rate: float = 1e-5, max_decisions: int = 24) -> dict:
        """One grouped ownplay update, normalized by the fixed decision horizon."""
        import torch
        if len(episodes) < 2 or any(not episode["replays"] for episode in episodes):
            raise ValueError("RL update requires at least two nonempty ownplay episodes")
        if max_decisions < 1 or any(len(ep["replays"]) > max_decisions for ep in episodes):
            raise ValueError("Replay exceeds the configured fixed decision horizon")
        rewards = torch.tensor([episode["reward"] for episode in episodes], dtype=torch.float32)
        if not torch.isfinite(rewards).all():
            raise ValueError("Nonfinite gameplay reward")
        advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
        if not advantages.abs().max().item():
            raise RuntimeError("Equal gameplay rewards give no learning signal; collect another rollout")
        before = self.adapter_hash()
        before_parameters = [param.detach().float().cpu().clone() for param in self.trainable]
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, max_replay_error = 0.0, 0.0
        for advantage, episode in zip(advantages.tolist(), episodes):
            for replay in episode["replays"]:
                new = self.logprob(replay)
                old = replay["old_logprob"].to(self.device)
                replay_error = abs(float((new.detach() - old).item()))
                max_replay_error = max(max_replay_error, replay_error)
                # Sampling executes denoising serially, replay uses block parallel
                # BF16 attention. Small numeric drift is expected, a different
                # policy or corrupted trace must fail before stepping optimizer.
                if not torch.isfinite(new) or replay_error > 0.05:
                    raise RuntimeError(f"Behavior density replay mismatch: {replay_error}")
                ratio = (new - old).exp()
                objective = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                loss = -objective / (len(episodes) * max_decisions)
                loss.backward()
                loss_sum += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.trainable, 1.0)
        if not math.isfinite(float(norm)) or float(norm) <= 0:
            raise RuntimeError(f"Invalid RL gradient norm: {float(norm)}")
        self.optimizer.step()
        after = self.adapter_hash()
        if after == before:
            raise RuntimeError("Optimizer did not change the action expert adapter")
        delta_l1 = sum(float((param.detach().float().cpu() - old).abs().sum())
                       for param, old in zip(self.trainable, before_parameters))
        return {"optimizer_updates": 1, "optimizer_steps": 1, "parameter_delta_l1": delta_l1, "loss": loss_sum, "gradient_norm": float(norm),
                "rewards": rewards.tolist(), "advantages": advantages.tolist(),
                "max_behavior_logprob_error": max_replay_error, "max_decisions": max_decisions,
                "adapter_sha256_before": before, "adapter_sha256_after": after}

    def save(self, output: Path) -> str:
        import torch
        output.mkdir(parents=True, exist_ok=True)
        self.model.expert.save_pretrained(output / "expert_adapter", safe_serialization=True)
        torch.save(self.optimizer.state_dict(), output / "optimizer.pt")
        manifest = {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                    "processor": PROCESSOR_ID, "processor_revision": PROCESSOR_REVISION,
                    "source_revisions": SOURCE_REVISIONS, "training": "ownplay_sde_policy_gradient",
                    "trainable_parameters": sum(p.numel() for p in self.trainable),
                    "adapter_sha256": self.adapter_hash(), "diffusion_steps": self.diffusion_steps,
                    "noise_level": self.noise_level, "converted_config": self.converted_config,
                    "buffer_restoration": self.buffer_restoration}
        (output / "policy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest["adapter_sha256"]
