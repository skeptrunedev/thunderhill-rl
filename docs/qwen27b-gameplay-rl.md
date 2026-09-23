# Qwen 27B gameplay RL validation

This path starts from the published Qwen/Qwen3.8-27B instruction checkpoint at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` with fresh LoRA adapters. It performs only simulator reward driven RL through the existing TRL Dr GRPO trajectory loss. It does not use OpenRouter outputs, demonstrations, supervised datasets or prior Gemma adapters as training targets.

Qwen native function calls use its pinned XML chat template and actual role tool feedback. XGrammar restricts syntax and integer ranges during sampling; the loss uses that same constrained probability distribution. Steering, throttle and brakes remain model decisions. The simulator supplies road telemetry and audits legal progress, failures, stalls and reward against recorded transitions.

The dedicated launcher is `training/modal_qwen_rl.py`. It requests one H100 with 80 GB, loads the language model in BF16, and adapts attention, DeltaNet projections and feed forward layers using rank 16 LoRA. Gradient checkpointing reduces training activation memory. Native convolution and DeltaNet kernels are required before weight allocation. Compiled inference and CUDA graph diagnostics remain enabled.

The bounded validation runs one generation with four sampled training episodes, up to 30 simulated seconds per episode. It evaluates two sampled attempts before and after the update plus greedy controls. These small evaluation groups validate the pipeline, not reliable driving improvement. GPU inference capacity and training microbatches are measured before use. A generation must produce finite nonzero gradients, changed adapter tensors, and matching saved/reloaded logits.

```bash
modal run --detach training/modal_qwen_rl.py --run-id qwen27b-h100-rl-01
```

The launch records source commit and file hashes. W&B uses the existing personal account through a private ephemeral Modal secret. Persistent artifacts are stored in the `thunderhill-runs-v2` volume under the run ID. The function limit is 7200 seconds; the child has a 6900 second limit with time reserved to flush recordings. No automatic cloud retry is enabled.

Every simulator attempt produces its recording and a video job, including probes and failures. Download the run with `modal volume get thunderhill-runs-v2 qwen27b-h100-rl-01 artifacts/qwen27b-h100-rl-01`; the existing local video queue renders the downloaded jobs. Hosted base model screening and local model training are separate experiments because provider precision, tool serialization and sampling differ.

CPU verification uses the real pinned tokenizer and a randomly initialized small Qwen hybrid architecture to verify native tool generation, sampling and loss likelihood agreement, adapter updates, frozen base weights and exact reload behavior. Synthetic rewards in these tests are plumbing tests only. Actual 27B GPU feasibility and performance must be established by the cloud result.
