# Experiment graphs

The current launcher enables NVIDIA Cosmos RL's console and W&B loggers. Prepared configuration uses project `thunderhill-rl` and experiment name `thunderhill-alpagym`. The native trainer owns metric names, update steps and logging. The previous custom trainer's flags, token metrics and generation axis are not the contract for this stack.

Use the same upstream Python environment as [training](../training/README.md) to authenticate:

```sh
/path/to/alpagym/.venv/bin/python -m wandb login
```

Select the owning account with the supported W&B environment configuration, including `WANDB_ENTITY` when needed. Keep credentials out of repository files and command arguments. For local logging without hosted uploads, set `WANDB_MODE=offline` before launching the prepared run. Set `WANDB_MODE=online` when hosted graphs are desired. Preparation alone does not start a W&B training run.

The authoritative experiment settings are the prepared resolved configuration and native worker logs. Inspect reward, policy loss, gradient behavior and optimizer steps alongside the simulator's episode summaries. A lower loss alone does not establish improved riding. Legal progress, crashes, offroad events, completion and legal lap times remain the gameplay measures that matter.

Each rollout worker also writes a separate gameplay diagnostics W&B run, grouped by the prepared run directory name. These charts report total reward, metric values and weighted reward components against actual policy version, with rollout and evaluation namespaces separated. Immutable batch reports are stored under `rollout_metrics` before the optional W&B write. This does not change NVIDIA's training reward, replay or native trainer logger.

Video generations identify Cosmos's actual policy weight version. This need not equal a dashboard row or optimizer minibatch number. Session identity joins the rollout's observations, trajectories, controls, reward summary and recording. Recordings and checkpoints remain separate artifacts; enabling scalar graphs does not by itself upload every video.

Full native model optimization and its hosted graph output still require a real GPU validation. See the official [W&B environment variable reference](https://docs.wandb.ai/models/track/environment-variables) for authentication and offline logging behavior.
