"""Explicit native checkpoint continuation and exact restored-state receipts.

This module does not implement optimization or create demonstrations. NVIDIA
loads its own checkpoint and remains responsible for every training update.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time


OPTIMIZER_SETTINGS = {
    "optimizer_lr": "optm_lr",
    "optimizer_warmup_steps": "optm_warmup_steps",
}


def load_resume_plan(
    parent: Path,
    *,
    checkpoint_step: int,
    settings: dict,
    optimizer_retune: bool = False,
) -> dict:
    """Read a stopped campaign without resetting its original wall deadline.

    Native Cosmos resume rebuilds the LambdaLR from the current config and then
    loads ``scheduler_rank_0.pth`` (``base_lrs``) and ``optimizer_rank_0.pth``
    (param group ``lr``/``initial_lr``). A changed learning rate would therefore
    be silently replaced by the checkpoint's. It is accepted only as an explicit
    optimizer retune, which ``verify_restored_state`` applies after proving the
    exact restore. Group size is never retunable: ``remain_samples_num`` counts
    rollouts, so reinterpreting it under another n_generation moves the data
    cursor and scheduler horizon.
    """
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
    candidates = [
        row for row in completed_exports(run_dir) if row["step"] == checkpoint_step
    ]
    if len(candidates) != 1:
        raise ValueError("Requested native checkpoint and export are not complete")
    checkpoint = Path(candidates[0]["checkpoint"])
    saved = json.loads((checkpoint / "cosmos_config").read_text())
    # Campaigns before these settings existed ran AlpaGym's defaults; the
    # checkpoint's own Cosmos config is the authority for what trained it.
    trained = {key: saved["train"][field] for key, field in OPTIMIZER_SETTINGS.items()}
    expected = {**trained, **identity["settings"]}
    if {key: expected[key] for key in trained} != trained:
        raise ValueError("Parent campaign settings disagree with its checkpoint")
    if settings.get("rollouts") != expected["rollouts"]:
        raise ValueError(
            f"Checkpoint group size is {expected['rollouts']} rollouts; its restored "
            "remain_samples_num counts rollouts of that size, so continuation "
            "cannot change n_generation. Start a fresh campaign for a new group size"
        )
    if set(settings) != set(expected) or any(
        settings[key] != expected[key] for key in expected if key not in trained
    ):
        raise ValueError("Continuation must preserve all original campaign settings")
    requested = {key: settings[key] for key in trained}
    retune = None
    if requested != trained:
        if not optimizer_retune:
            raise ValueError(
                f"Checkpoint was trained with {trained} but {requested} was requested. "
                "Native resume restores the checkpoint scheduler and optimizer "
                "learning rate, so the new value would be silently ignored; pass "
                "an explicit optimizer retune to apply it after exact restore"
            )
        retune = {"from": trained, "to": requested}
    elif optimizer_retune:
        raise ValueError("Optimizer retune requested without a changed optimizer setting")
    deadline = float(identity["hard_deadline_unix"])
    if deadline - time.time() <= 3600:
        raise ValueError(
            "Original campaign deadline leaves insufficient continuation time"
        )
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
        optimizer_retune=retune,
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
    if previous.get("optimizer_retune") != resume_plan["optimizer_retune"]:
        raise ValueError("Retry differs from the failed continuation's optimizer retune")
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
    retune = plan.get("optimizer_retune")
    for item, values in ((saved, "from"), (actual, "to")):
        for key in ("resume", "output_dir", "timestamp"):
            item["train"].pop(key, None)
        item["train"]["train_policy"].pop("trainer_type", None)
        if retune is not None:
            # Only the declared optimizer fields may differ, and only to the
            # exact values the continuation plan recorded.
            for setting, field in OPTIMIZER_SETTINGS.items():
                if item["train"].pop(field) != retune[values][setting]:
                    raise ValueError(
                        f"Native {field} differs from the planned optimizer retune"
                    )
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


def apply_optimizer_retune(schedulers, learning_rate: float) -> dict:
    """Move a verified restored LambdaLR onto a new base learning rate.

    Cosmos builds each LambdaLR from the current config, so its lambda already
    carries the new warmup/decay shape; only ``base_lrs`` and the optimizer
    param group ``initial_lr``/``lr`` come from the checkpoint. Step count and
    Adam moments are kept. The next optimizer step uses the returned rate.
    """
    before, after = [], []
    for scheduler in schedulers:
        groups = scheduler.optimizer.param_groups
        if len(scheduler.base_lrs) != len(groups) or len(
            scheduler.lr_lambdas
        ) != len(groups):
            raise ValueError("Restored scheduler does not match its optimizer groups")
        before.extend(group["lr"] for group in groups)
        scheduler.base_lrs = [learning_rate] * len(groups)
        for group, factor in zip(groups, scheduler.lr_lambdas):
            group["initial_lr"] = learning_rate
            group["lr"] = learning_rate * factor(scheduler.last_epoch)
        scheduler._last_lr = [group["lr"] for group in groups]
        after.extend(scheduler.get_last_lr())
    return dict(
        target_base_lr=learning_rate,
        restored_lrs=sorted(set(before)),
        applied_lrs=sorted(set(after)),
        applied=bool(after) and all(value > 0 for value in after),
    )


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
    retune = plan.get("optimizer_retune")
    learning_rate = None
    if all(checks.values()):
        # CheckpointMananger already rebuilt this scheduler using the saved
        # horizon and loaded its state before restoring the optimizer. Native
        # GRPO's first-batch rebuild would reset optimizer LR to warmup zero.
        # Retain the verified native scheduler instead of initializing it twice.
        trainer.lr_schedulers_updated = True
        if retune is not None:
            learning_rate = apply_optimizer_retune(
                trainer.lr_schedulers, retune["to"]["optimizer_lr"]
            )
            checks["optimizer_retune_applied"] = learning_rate["applied"]
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        checkpoint=str(checkpoint),
        checkpoint_step=plan["checkpoint_step"],
        fallback_to_base_accepted=False,
        native_restored_scheduler_initialized=all(checks.values()),
        optimizer_retune=retune,
        learning_rate=learning_rate,
    )
    from training.native_campaign import write_json

    write_json(run_dir / "resume_verification.json", report)
    if not report["passed"]:
        raise RuntimeError(
            "Native checkpoint state did not restore exactly: " + str(checks)
        )
    return report
