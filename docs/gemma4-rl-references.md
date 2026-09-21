# Gemma 4 E4B RL references

Audited September 19, 2026. Requirements: train Gemma 4 E4B itself, use actual environment rewards, support repeated actions and observations, target parallel rollouts on one RTX PRO 6000 96GB, and preserve Harbor integration. No repository found demonstrated this complete combination for racing. These are source audits, not reproduced GPU training runs.

## Exact model training reference

[lamm-mit/graph-preflexor-grpo](https://github.com/lamm-mit/graph-preflexor-grpo/tree/e267639dbf2d3b2312e8d79dcecd2fb8c36613f4) has an explicit Gemma 4 E4B supervised warm start followed by GRPO with LoRA.

Read [the E4B runbook](https://github.com/lamm-mit/graph-preflexor-grpo/blob/e267639dbf2d3b2312e8d79dcecd2fb8c36613f4/GEMMA4_LAMBDA_NOTES.md) and [the trainer](https://github.com/lamm-mit/graph-preflexor-grpo/blob/e267639dbf2d3b2312e8d79dcecd2fb8c36613f4/src/run_grpo_graph.py). The trainer constructs GRPOConfig, attaches trainable adapters, supplies reward functions, invokes GRPOTrainer.train, and saves checkpoints. It exposes generation count, batching, accumulation, and optional colocated or server vLLM generation.

Limits: graph reasoning rather than an interactive game; rewards include an external language model judge. The current run command in the runbook does not enable its optional vLLM flag. Single H100 recommendations are author guidance, not our measurements. Three referenced Hugging Face model endpoints returned HTTP 401 without authentication, so public checkpoint availability was not verified. Do not claim a reproduced E4B run or verified RTX performance.

## Closest Gemma game example

[Unsloth's 2048 script](https://github.com/unslothai/notebooks/blob/bce1b9d12e566c9df217fa236904ae3da3415247/python_scripts/Gemma4_%28E2B%29_Reinforcement_Learning_2048_Game.py) loads Gemma 4 E2B, attaches LoRA adapters, and trains through TRL GRPO. Its reward executes generated strategy code in an actual game, sharing the game seed among candidates in a reward call.

Important distinction: the language model generates a Python strategy function. That function chooses moves during the game. The LLM does not receive a new observation and select each move. This is useful for executable outcome rewards, but it is not the direct motorcycle control loop we want. It also uses E2B rather than E4B and sets fast_inference=False. Changing the model name alone would not validate our workload.

## Best environment architecture reference

[Hugging Face TRL](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md) documents environment_factory with separate environment instances, reset, public action tools, and an episode get_reward. Tools can return images for subsequent generation turns. The tested model list includes Gemma 4 E2B. It also documents Harbor integration through the same environment interface.

This is the closest architecture to our requirement: observe, generate controls, advance Godot by explicit simulation steps, observe again, and update the language model using the resulting episode reward. Exact E4B compatibility, the Harbor adapter, and efficient generation on our GPU still require an executed integration test. Use a shared reset key to compare candidates from identical initial simulator states.

## Accelerated generation support is framework specific

[NeMo RL's Gemma guide](https://docs.nvidia.com/nemo/rl/nightly/guides/models/gemma/gemma4.html) provides an exact E4B visual GRPO recipe with vLLM. Its published recipe targets multiple GPUs and geometry questions, rather than a single GPU racing game. This means a blanket claim that Gemma RL cannot use vLLM is too broad. Unsloth's documented limitation applies to that integration path, not every framework.

## Decision

Use the concrete TRL Gemma CARLA and Catch examples below as the primary control references, and the Graph PRefLexOR repository for exact E4B adapter training details. Unsloth 2048 remains a supplementary game outcome reward example. Before long training, verify nonzero adapter updates, action dependent rewards, model synchronization after updates, peak memory, and sustained batched rollout throughput on the intended GPU. No concurrency or throughput number is established yet.

## Selected direct control references

Personally inspected both complete scripts at TRL commit `a98fa6a4428f9aae58dfb26d729d7437f662f27a`.

### Gemma CARLA driving

[carla_vlm_gemma.py](https://github.com/huggingface/trl/blob/a98fa6a4428f9aae58dfb26d729d7437f662f27a/examples/grpo_carla/carla_vlm_gemma.py) is the closest concrete driving example found. It defaults to `google/gemma-4-E2B-it`, offers LoRA, exposes observe, emergency_stop, and lane_change tools, advances simulation ticks, and returns camera images and vehicle descriptions after actions. It passes each environment's resulting rubric reward to GRPOTrainer and calls trainer.train(). The LLM selects actions repeatedly, rather than generating a separate controller program.

Limits: an emergency obstacle avoidance scenario, not racing or continuous motorcycle controls. The source requires at least two CARLA server URLs, with one concurrent connection per server. Its configuration does not enable vLLM. Its LoRA exclusions express the intention to omit vision components; actual trainable parameter names must be checked on E4B. We have inspected the code but not executed this training run.

### Catch with colocated vLLM

[grpo_catch.py](https://github.com/huggingface/trl/blob/a98fa6a4428f9aae58dfb26d729d7437f662f27a/examples/grpo_catch/grpo_catch.py) exposes move and stay actions. Each action steps OpenSpiel and returns an updated observation; completed catch outcomes supply GRPO rewards. The script documents a single GPU colocated vLLM mode and configures multiple candidate generations, training batches, and accumulation.

Limits: it defaults to Qwen2.5 0.5B, uses text observations, and does not establish E4B throughput or adapter synchronization. Its candidate count is a configuration setting, not a measured number of simultaneous Godot simulations. For our environment, verify session isolation and identical reset states within each candidate group.

### Combination for Thunderhill

| Required component | Reference and intended adaptation |
| --- | --- |
| Gemma 4 E4B adapter training | Graph PRefLexOR's exact model configuration and TRL LoRA setup. Verify target modules on the loaded model. |
| Repeated driving observations and actions | Gemma CARLA wrapper. Replace its emergency tools with validated motorcycle control inputs and fixed simulation steps. |
| Batched generation on one GPU | Catch's colocated vLLM path. Establish E4B support, memory use, and correct weight updates before scaling. |
| Racing reward | Godot trajectory outcomes: legal progress, completed laps, lap time, and crashes. Do not reuse graph judge rewards or emergency scenario rubrics. |
| Harbor integration | Preserve the project's Harbor task and verifier requirement. CARLA and Catch use OpenEnv, so they supply control patterns, not a completed Harbor adapter. |
| Recording | Save policy checkpoint identifiers, initial state, actions, and state trajectories for consistent evaluation replays. |

First integration acceptance: two isolated simulator sessions from a shared initial state produce action dependent outcomes; the trainer updates E4B adapters; the rollout model receives those updates; held out evaluation and replay artifacts identify the exact checkpoint. Scale rollout counts only after measuring this loop on the intended RTX PRO. This is an implementation specification, not a claim that the integration is built.

## September 21 executed local validation

Selected TRL as the primary framework after rechecking the current CARLA and
Catch examples and its Harbor interface. An actual RTX 2080 Ti run using TRL
1.13.0, Gemma 3 270M and PEFT passed the small launch probe: game dependent
rewards, nonzero adapter updates, changed policy logits, checkpoint reload and
recorded episode attribution. [Reproduction and exact scope](../training/README.md).
This does not validate E4B or repeated camera control.

The earlier unauthenticated checkpoint access limitation is now outdated for the
official model: `google/gemma-4-E4B-it` API and config requests succeeded, with
`gated=false` and revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
No E4B weights were downloaded or trained during this local validation.
