"""Explicit native checkpoint continuation and exact restored-state receipts.

This module does not implement optimization or create demonstrations. NVIDIA
loads its own checkpoint and remains responsible for every training update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


def load_resume_plan(parent: Path, *, checkpoint_step: int, settings: dict) -> dict:
    """Read a stopped campaign without resetting its original wall deadline."""
    from training.native_campaign import completed_exports

    parent = parent.resolve()
    campaign = json.loads((parent / "campaign.json").read_text())
    identity = json.loads((parent / "source_identity.json").read_text())
    run_dir = parent / Path(campaign["run_dir"]).name
    status = json.loads((run_dir / "run_status.json").read_text())
    if status.get("runtime_stopped") is not True:
        raise ValueError(
            "The parent runtime must finish owned cleanup before continuation"
        )
    expected = identity["settings"]
    if settings != expected:
        raise ValueError("Continuation must preserve all original campaign settings")
    deadline = float(identity["hard_deadline_unix"])
    if deadline - time.time() <= 3600:
        raise ValueError(
            "Original campaign deadline leaves insufficient continuation time"
        )
    candidates = [
        row for row in completed_exports(run_dir) if row["step"] == checkpoint_step
    ]
    if len(candidates) != 1:
        raise ValueError("Requested native checkpoint and export are not complete")
    checkpoint = Path(candidates[0]["checkpoint"])
    saved = json.loads((checkpoint / "cosmos_config").read_text())
    if saved["train"]["train_policy"]["kl_beta"] != 0:
        raise ValueError(
            "Native checkpoints omit the KL reference model; exact continuation requires zero KL"
        )
    if saved["train"]["max_num_steps"] != settings["max_steps"]:
        raise ValueError("Continuation cannot change the native scheduler horizon")
    import torch

    extra = torch.load(
        checkpoint / "extra_info_rank_0.pth", map_location="cpu", weights_only=False
    )
    if (
        extra.get("step") != checkpoint_step
        or extra.get("total_steps") != settings["max_steps"]
        or not {"torch", "numpy", "python", "cuda"} <= extra.get("rng_state", {}).keys()
        or not isinstance(extra.get("remain_samples_num"), int)
    ):
        raise ValueError(
            "Native checkpoint lacks matching step, scheduler, RNG or data cursor state"
        )
    previous_resume = campaign.get("native_resume")
    baseline = (
        Path(previous_resume["baseline_evaluation"])
        if previous_resume
        else run_dir / "recordings/baseline/evaluation.json"
    )
    if not baseline.is_file():
        raise ValueError("Original matched baseline is missing")
    return dict(
        schema="native_checkpoint_continuation_v1",
        parent_campaign=str(parent),
        parent_run=str(run_dir),
        checkpoint=str(checkpoint),
        checkpoint_step=checkpoint_step,
        export=candidates[0]["export"],
        hard_deadline_unix=deadline,
        deadline_unix=float(campaign["deadline_unix"]),
        baseline_evaluation=str(baseline),
        baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        checkpoint_config_sha256=hashlib.sha256(
            (checkpoint / "cosmos_config").read_bytes()
        ).hexdigest(),
        extra_info_sha256=hashlib.sha256(
            (checkpoint / "extra_info_rank_0.pth").read_bytes()
        ).hexdigest(),
        settings=settings,
        kl_beta=0.0,
        remain_samples_num=extra["remain_samples_num"],
    )


def register_verified_resume_trainer():
    from alpagym_runtime.cosmos.trainer import AlpagymGRPOTrainer
    from cosmos_rl.policy.trainer.base import TrainerRegistry

    @TrainerRegistry.register("thunderhill_verified_native_resume")
    class VerifiedNativeResume(AlpagymGRPOTrainer):
        def weight_resume(self):
            # Native GRPO catches load errors and can fall back to base weights.
            # Verification outside its catch makes that fallback fatal here.
            verify_resume_configuration(self.config)
            extra = super().weight_resume()
            verify_restored_state(self, extra)
            return extra

    return VerifiedNativeResume


def verify_failed_continuation_retry(failed_campaign: Path, resume_plan: dict) -> dict:
    """Permit an explicit successor only after a failed attempt made no update."""
    from training.native_campaign import summarize

    failed_campaign = failed_campaign.resolve()
    allocation_path = failed_campaign / "allocation_status.json"
    allocation = json.loads(allocation_path.read_text())
    campaign = json.loads((failed_campaign / "campaign.json").read_text())
    identity = json.loads((failed_campaign / "source_identity.json").read_text())
    previous = identity["native_resume"]
    for key in (
        "parent_campaign",
        "checkpoint",
        "checkpoint_step",
        "hard_deadline_unix",
        "deadline_unix",
        "settings",
    ):
        if previous[key] != resume_plan[key]:
            raise ValueError(
                "Retry differs from the failed continuation's original lineage or budget"
            )
    run_dir = failed_campaign / Path(campaign["run_dir"]).name
    status = json.loads((run_dir / "run_status.json").read_text())
    if (
        allocation.get("state") != "failed"
        or not allocation.get("finished_at_unix")
        or campaign.get("state") != "failed"
        or status.get("runtime_stopped") is not True
    ):
        raise ValueError(
            "Only a terminal failed allocation with completed runtime cleanup can be retried"
        )
    progress = summarize(run_dir)
    if (
        progress["trainer_steps_started"]
        or progress["nonzero_gradient_log_records"]
        or progress["completed_native_checkpoints"]
        or any(run_dir.glob("cosmos/**/checkpoints/step_*"))
    ):
        raise ValueError(
            "Failed continuation reached training; refusing to restart its original checkpoint"
        )
    policy_logs = [
        path
        for path in (run_dir / "logs").rglob("*.log")
        if "verify_restored_state" in path.read_text(errors="replace")
    ]
    if not policy_logs:
        raise ValueError(
            "Retry requires evidence of failure in checkpoint verification before training"
        )
    return dict(
        failed_campaign=failed_campaign.name,
        failed_directory=str(failed_campaign),
        allocation_sha256=hashlib.sha256(allocation_path.read_bytes()).hexdigest(),
        runtime_stopped=True,
        trainer_steps_started=[],
        new_checkpoints=0,
        original_deadline_unix=resume_plan["hard_deadline_unix"],
    )


def verify_resume_configuration(config) -> None:
    """Require identical optimizer, schedule, objective and model geometry."""
    run_dir = Path(config.custom["resolved_config_path"]).parent
    plan = json.loads((run_dir / "resume_plan.json").read_text())
    saved = json.loads((Path(plan["checkpoint"]) / "cosmos_config").read_text())
    actual = json.loads(config.model_dump_json())
    for item in (saved, actual):
        for key in ("resume", "output_dir", "timestamp"):
            item["train"].pop(key, None)
        item["train"]["train_policy"].pop("trainer_type", None)
    for section in ("train", "policy", "rollout"):
        if saved[section] != actual[section]:
            raise ValueError(f"Continuation changed native {section} configuration")


def exact_resume_state_equal(left, right) -> bool:
    """Compare both checkpoint and live DTensors without distributed dispatch."""
    import numpy as np
    import torch
    from torch.distributed.tensor import DTensor

    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.dtype != right.dtype
            or left.shape != right.shape
        ):
            return False
        left_distributed, right_distributed = (
            isinstance(left, DTensor),
            isinstance(right, DTensor),
        )
        if left_distributed != right_distributed:
            return False
        if left_distributed:
            if (
                left.device_mesh.size() != 1
                or right.device_mesh.size() != 1
                or left.placements != right.placements
                or left.device_mesh.mesh_dim_names != right.device_mesh.mesh_dim_names
                or not torch.equal(
                    left.device_mesh.mesh.cpu(), right.device_mesh.mesh.cpu()
                )
            ):
                return False
            left, right = left.to_local(), right.to_local()
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(exact_resume_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(exact_resume_state_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    return type(left) is type(right) and left == right


def verify_restored_state(trainer, extra: dict) -> dict:
    import torch
    from cosmos_rl.utils.checkpoint import CheckpointMananger

    run_dir = Path(trainer.config.custom["resolved_config_path"]).parent
    plan = json.loads((run_dir / "resume_plan.json").read_text())
    checkpoint = Path(plan["checkpoint"])
    if trainer.parallel_dims.world_size != 1 or trainer.global_rank != 0:
        raise ValueError(
            "Exact resume verification requires the qualified single policy rank"
        )
    if Path(str(trainer.config.train.resume)).resolve() != checkpoint.resolve():
        raise ValueError(
            "Native resume path differs from the explicit continuation plan"
        )
    if trainer.config.train.train_policy.kl_beta != 0:
        raise ValueError("Resume cannot reconstruct an unsaved KL reference model")
    for name, field in (
        ("cosmos_config", "checkpoint_config_sha256"),
        ("extra_info_rank_0.pth", "extra_info_sha256"),
    ):
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != plan[field]:
            raise ValueError("Native resume metadata changed after preparation")
    saved_extra = torch.load(
        checkpoint / "extra_info_rank_0.pth", map_location="cpu", weights_only=False
    )
    expected_extra = {
        key: value for key, value in saved_extra.items() if key != "rng_state"
    }
    checks = {
        "extra_state_exact": exact_resume_state_equal(expected_extra, extra),
        "step_exact": extra.get("step") == plan["checkpoint_step"],
        "scheduler_horizon_exact": extra.get("total_steps")
        == plan["settings"]["max_steps"],
        "policy_rng_exact": exact_resume_state_equal(
            saved_extra["rng_state"], CheckpointMananger.get_rng_state()
        ),
    }
    # Memory-map saved tensors and compare one tensor at a time. This avoids
    # retaining another complete model/optimizer snapshot in host memory.
    for name, current in (
        ("model", trainer.model),
        ("optimizer", trainer.optimizers),
        ("scheduler", trainer.lr_schedulers),
    ):
        saved = torch.load(
            checkpoint / f"{name}_rank_0.pth",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        checks[name + "_exact"] = exact_resume_state_equal(saved, current.state_dict())
        del saved
    if all(checks.values()):
        # CheckpointMananger already rebuilt this scheduler using the saved
        # horizon and loaded its state before restoring the optimizer. Native
        # GRPO's first-batch rebuild would reset optimizer LR to warmup zero.
        # Retain the verified native scheduler instead of initializing it twice.
        trainer.lr_schedulers_updated = True
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        checkpoint=str(checkpoint),
        checkpoint_step=plan["checkpoint_step"],
        fallback_to_base_accepted=False,
        native_restored_scheduler_initialized=all(checks.values()),
    )
    from training.native_campaign import write_json

    write_json(run_dir / "resume_verification.json", report)
    if not report["passed"]:
        raise RuntimeError(
            "Native checkpoint state did not restore exactly: " + str(checks)
        )
    return report
