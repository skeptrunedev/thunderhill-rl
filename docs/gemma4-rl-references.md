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

Use TRL as the primary environment integration reference, the Graph PRefLexOR repository for exact E4B adapter training details, and Unsloth 2048 for an inspectable game outcome reward example. Replace graph and code generation rewards with actual driving outcomes. Before long training, verify nonzero adapter updates, action dependent rewards, model synchronization after updates, peak memory, and sustained batched rollout throughput on the intended GPU. No concurrency or throughput number is established yet.
