"""Full episode policy gradients using TRL 1.13.0's Dr GRPO token loss.

Each action is replayed against its original stateless prompt. One optimizer
step follows all actions in a generation. The fixed episode horizon normalizer
never rewards early termination by increasing its per token gradient weight.
"""
from dataclasses import dataclass, replace
import json
import math
import time
from pathlib import Path

import torch
import trl
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from model_runtime import inference_precision


@dataclass(frozen=True)
class TrajectoryConfig:
    max_actions: int
    microbatch_size: int = 1
    learning_rate: float = 1e-5
    max_completion_length: int = 32
    max_grad_norm: float = 1.0


class _TrajectoryLossTrainer(GRPOTrainer):
    """Audit the same likelihood tensor used by TRL, without a second forward."""

    def trajectory_loss(self, model, batch):
        self._trajectory_reference = (batch["old_per_token_logps"], batch["completion_mask"])
        self.behavior_logp_difference = 0.0
        try:
            return super().compute_loss(model, batch)
        finally:
            self._trajectory_reference = None

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
        reference = getattr(self, "_trajectory_reference", None)
        if reference is not None:
            old, mask = reference
            with torch.no_grad():
                self.behavior_logp_difference = (result[0].detach() - old).abs()[mask.bool()].max().item()
        return result


class TrajectoryUpdater:
    def __init__(self, model, tokenizer, output_dir, config: TrajectoryConfig):
        if trl.__version__ != "1.13.0":
            raise RuntimeError("Trajectory loss integration requires TRL 1.13.0")
        if min(config.max_actions, config.microbatch_size, config.max_completion_length) < 1:
            raise ValueError("Action limits and microbatch size must be positive")
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
            raise ValueError("Gradient norm limit must be finite and positive")
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        if not self.parameters:
            raise ValueError("No trainable parameters")
        # Never cast the adapter with the base model. BF16 base plus FP32 LoRA is
        # the established runtime; optimizer state follows each parameter dtype.
        self.parameter_dtypes = [p.dtype for p in self.parameters]
        args = GRPOConfig(
            output_dir=str(self.output_dir / "trl"),
            per_device_train_batch_size=2, num_generations=2,
            gradient_accumulation_steps=1, steps_per_generation=1,
            max_completion_length=config.max_completion_length,
            loss_type="dr_grpo", scale_rewards="none", beta=0.0,
            temperature=1.0, top_p=1.0, top_k=0,
            gradient_checkpointing=False, bf16=False, fp16=False,
            use_cpu=model.device.type == "cpu", report_to="none",
            save_strategy="no", disable_dropout=True,
        )
        self.trainer = _TrajectoryLossTrainer(
            model=model, args=args, processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"prompt": ["unused", "unused"]}),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
        )
        self.trainer.current_gradient_accumulation_steps = 1
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate, weight_decay=0.0)

    def rows(self, episodes):
        if len(episodes) < 2:
            raise ValueError("At least two episodes are required for group advantages")
        rewards = [float(e["reward"] if "reward" in e else e["reward_components"]["total"]) for e in episodes]
        if not all(math.isfinite(r) for r in rewards):
            raise ValueError("Nonfinite episode reward")
        mean = sum(rewards) / len(rewards)
        rows = []
        for episode_index, (episode, reward) in enumerate(zip(episodes, rewards, strict=True)):
            decisions = episode["decisions"]
            if not 1 <= len(decisions) <= self.config.max_actions:
                raise ValueError("Episode action count outside configured horizon")
            for action_index, decision in enumerate(decisions):
                prompt = list(decision["prompt_ids"])
                completion = list(decision["completion_ids"])
                old = list(decision["old_per_token_logps"])
                if not prompt or not 1 <= len(completion) <= self.config.max_completion_length:
                    raise ValueError("Missing prompt or invalid action token count")
                if len(old) != len(completion) or not all(math.isfinite(v) for v in old):
                    raise ValueError("Every generated token needs a finite behavior log probability")
                if self.tokenizer.pad_token_id in completion and self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                    raise ValueError("Completion contains padding; provide only sampled tokens")
                eos = self.tokenizer.eos_token_id
                if eos in completion[:-1]:
                    raise ValueError("Completion contains tokens after EOS")
                if completion[-1] != eos and len(completion) != self.config.max_completion_length:
                    raise ValueError("Completion stopped before EOS without reaching token limit")
                rows.append(dict(prompt_ids=prompt, completion_ids=completion,
                                 old_per_token_logps=old, advantage=reward - mean,
                                 episode_index=episode_index, action_index=action_index))
        return rows, rewards

    def batch(self, rows):
        device = self.model.device
        pmax = max(len(r["prompt_ids"]) for r in rows)
        cmax = max(len(r["completion_ids"]) for r in rows)
        pad = self.tokenizer.pad_token_id
        result = {key: [] for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "old_per_token_logps", "advantages")}
        for row in rows:
            npad, ncomp = pmax - len(row["prompt_ids"]), len(row["completion_ids"])
            result["prompt_ids"].append([pad] * npad + row["prompt_ids"])
            result["prompt_mask"].append([0] * npad + [1] * (pmax - npad))
            result["completion_ids"].append(row["completion_ids"] + [pad] * (cmax - ncomp))
            result["completion_mask"].append([1] * ncomp + [0] * (cmax - ncomp))
            result["old_per_token_logps"].append(row["old_per_token_logps"] + [0.0] * (cmax - ncomp))
            result["advantages"].append(row["advantage"])
        return {k: torch.tensor(v, device=device, dtype=torch.float32 if k in ("old_per_token_logps", "advantages") else torch.long) for k, v in result.items()}

    def profile_microbatches(self, episodes, candidates=(1, 2, 4, 8, 16, 32)):
        """Measure actual TRL forward/backward shapes; never take an optimizer step.

        Probes repeat actual sampled rows when necessary. Include the longest
        prompt and completion to exercise their combined padding. Reserve memory
        for AdamW moments before they exist, plus 15 percent total device slack.
        """
        rows, _ = self.rows(episodes)
        if not candidates or any(not isinstance(n, int) or n < 1 for n in candidates):
            raise ValueError("Profiler candidates must be positive integers")
        device = self.model.device
        cuda = device.type == "cuda"
        was_training = self.model.training
        trainable_bytes = sum(p.numel() * p.element_size() for p in self.parameters)
        resident_state_bytes = sum(
            value.numel() * value.element_size()
            for state in self.optimizer.state.values() for value in state.values()
            if isinstance(value, torch.Tensor) and value.device == device
        )
        optimizer_reserve = max(0, 2 * trainable_bytes - resident_state_bytes)
        ordered = sorted(rows, key=lambda row: len(row["prompt_ids"]) + len(row["completion_ids"]), reverse=True)
        longest_prompt = max(rows, key=lambda row: len(row["prompt_ids"]))
        longest_completion = max(rows, key=lambda row: len(row["completion_ids"]))
        results = []
        try:
            self.model.train()
            for size in sorted(set(candidates)):
                self.optimizer.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)
                    free, total = torch.cuda.mem_get_info(device)
                    external_bytes = max(0, total - free - torch.cuda.memory_reserved(device))
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    total, external_bytes = 0, 0
                chosen = [ordered[i % len(ordered)] for i in range(size)]
                if size >= 2:
                    chosen[:2] = [longest_prompt, longest_completion]
                batch = None
                loss = None
                try:
                    batch = self.batch(chosen)
                    # Warm the kernels once, then measure two real backwards.
                    timings = []
                    for iteration in range(3):
                        self.optimizer.zero_grad(set_to_none=True)
                        if cuda:
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        with inference_precision(self.model):
                            loss = self.trainer.trajectory_loss(self.model, batch)
                            loss = loss * size / (len(episodes) * self.config.max_actions)
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite profiler loss")
                        loss.backward()
                        if cuda:
                            torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - started
                        if iteration:
                            timings.append(elapsed)
                    elapsed = sum(timings) / len(timings)
                    peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else None
                    peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else None
                    projected = max(peak_allocated, peak_reserved) + external_bytes + optimizer_reserve if cuda else None
                    safe = projected <= total * 0.85 if cuda else True
                    results.append(dict(
                        microbatch_size=size, seconds=elapsed,
                        actions_per_second=size / elapsed,
                        peak_allocated_bytes=peak_allocated,
                        peak_reserved_bytes=peak_reserved,
                        external_device_bytes=external_bytes,
                        projected_with_optimizer_bytes=projected,
                        device_total_bytes=total if cuda else None,
                        fits_with_headroom=safe,
                    ))
                    if not safe:
                        break
                except torch.cuda.OutOfMemoryError as error:
                    results.append(dict(microbatch_size=size, fits_with_headroom=False, error=type(error).__name__))
                    break
                finally:
                    loss, batch = None, None
                    self.optimizer.zero_grad(set_to_none=True)
                    if cuda:
                        torch.cuda.empty_cache()
            safe_results = [r for r in results if r["fits_with_headroom"]]
            if not safe_results:
                raise RuntimeError("No training microbatch fits with required GPU headroom")
            best = max(safe_results, key=lambda row: row["actions_per_second"])
            self.config = replace(self.config, microbatch_size=best["microbatch_size"])
            report = dict(
                selected_microbatch_size=best["microbatch_size"], headroom_fraction=0.15,
                optimizer_state_reserve_bytes=optimizer_reserve,
                probes=results, optimizer_steps=0,
                sampled_rows=len(rows), repeated_actual_rows_for_capacity=True,
            )
            return report
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def accumulate(self, episodes):
        """Accumulate one complete generation, without mutating parameters."""
        rows, rewards = self.rows(episodes)
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        loss_total = 0.0
        max_logp_difference = 0.0
        tokens_seen = 0
        started = time.monotonic()
        for offset in range(0, len(rows), self.config.microbatch_size):
            chunk = rows[offset:offset + self.config.microbatch_size]
            batch = self.batch(chunk)
            with inference_precision(self.model):
                loss = self.trainer.trajectory_loss(self.model, batch)
                max_logp_difference = max(max_logp_difference, self.trainer.behavior_logp_difference)
                loss = loss * (len(chunk) / (len(episodes) * self.config.max_actions))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite trajectory loss")
            loss.backward()
            loss_total += loss.detach().item()
            tokens_seen += int(batch["completion_mask"].sum())
            if offset // self.config.microbatch_size % 100 == 0 or offset + len(chunk) == len(rows):
                progress = dict(event="training_progress", actions_trained=offset + len(chunk),
                                actions_total=len(rows), tokens_trained=tokens_seen,
                                elapsed_seconds=time.monotonic() - started)
                temporary = self.output_dir / "progress.pending"
                temporary.write_text(json.dumps(progress) + "\n")
                temporary.replace(self.output_dir / "progress.json")
                print(json.dumps(progress), flush=True)
        expected_tokens = sum(len(r["completion_ids"]) for r in rows)
        if tokens_seen != expected_tokens:
            raise AssertionError("Not all generated tokens received a loss mask")
        return {
            "method": "TRL Dr GRPO full episode stateless action replay",
            "trl_version": trl.__version__, "loss": loss_total,
            "episodes": len(episodes), "actions": len(rows), "generated_tokens": expected_tokens,
            "trained_tokens": tokens_seen, "later_actions": sum(r["action_index"] > 0 for r in rows),
            "eos_tokens": sum(r["completion_ids"][-1] == self.tokenizer.eos_token_id for r in rows),
            "prompt_tokens_in_loss": 0, "padding_tokens_in_loss": 0,
            "fixed_normalizer": len(episodes) * self.config.max_actions * self.config.max_completion_length,
            "max_behavior_logp_difference": max_logp_difference,
            "rewards": rewards, "advantages": [r - sum(rewards) / len(rewards) for r in rewards],
            "optimizer_steps": 0,
        }

    def update(self, episodes, generation=0):
        path = self.output_dir / f"generation-{generation:04d}-update.json"
        if path.exists():
            raise FileExistsError(f"Generation already updated: {path}")
        before = [p.detach().cpu().clone() for p in self.parameters]
        audit = self.accumulate(episodes)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.config.max_grad_norm, error_if_nonfinite=True)
        if not torch.isfinite(norm) or norm.item() == 0:
            raise ValueError("Trajectory update has no finite nonzero gradient")
        self.optimizer.step()
        delta = 0.0
        for parameter, previous, dtype in zip(self.parameters, before, self.parameter_dtypes, strict=True):
            if parameter.dtype != dtype or not torch.isfinite(parameter).all():
                raise ValueError("Parameter precision changed or update became nonfinite")
            delta += (parameter.detach().cpu().float() - previous.float()).abs().sum().item()
        if not math.isfinite(delta) or delta == 0:
            raise ValueError("Trajectory update did not change trainable parameters")
        audit.update(generation=generation, gradient_norm=norm.item(), parameter_delta_l1=delta, optimizer_steps=1)
        with path.open("x") as stream:
            json.dump(audit, stream, indent=2)
            stream.write("\n")
        return audit
