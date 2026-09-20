# Small language models for racing

Source audit on September 19, 2026. Code inspection and published artifacts establish precedents; training results below have not been independently reproduced.

## Closest firm racing reference: RobotxR1

[ForzaETH/LLMxRobot](https://github.com/ForzaETH/LLMxRobot) accompanies RobotxR1 and includes supervised fine tuning followed by GRPO for Qwen2.5 1.5B and 3B. The [1.5B MPC adapter](https://huggingface.co/nibauman/MPCxR1_Qwen1.5B_SFT_GRPO) is published. The repository documents integration with the ForzaETH racing stack.

In [rl_mpc_train.py](https://github.com/ForzaETH/LLMxRobot/blob/main/train/rl_mpc_train.py), generated controller parameters are applied through ROS, subsequent driving is evaluated, and the resulting error becomes a reward for GRPOTrainer. This updates language model adapters using environment feedback.

The [driving evaluator](https://github.com/ForzaETH/LLMxRobot/blob/main/train/utils/mpc/eval_driving.py) includes racing line tracking, centerline tracking, speed tracking, reversing, and smoothness. Racing line tracking follows an existing line. The LLM adjusts an existing model predictive controller; it does not directly discover the fastest lap or produce motorcycle steering and throttle commands. Candidate evaluation resets controller parameters, but does not establish identical full simulator state before every candidate.

Use this as the strongest racing related reference for the training loop and released small model artifacts. Its published performance improvements must not be described as lap time gains.

## Smaller language backbone: MindDrive

[xiaomi-mlab/MindDrive](https://github.com/xiaomi-mlab/MindDrive) provides imitation learning followed by PPO in CARLA, with checkpoints and evaluation result files. Its [training configuration](https://github.com/xiaomi-mlab/MindDrive/blob/main/adzoo/minddrive/configs/minddrive_rl_ppo_train.py) uses a Qwen2 0.5B language backbone plus a separate EVAViT vision encoder and additional perception and planning components. The whole system is larger than 0.5B parameters.

The [model implementation](https://github.com/xiaomi-mlab/MindDrive/blob/main/mmcv/models/detectors/minddrive.py) uses a decision expert adapter for PPO and an action expert to translate decisions into trajectories. This is evidence for RL updates to a small language backbone in driving, but it is urban driving rather than racing. Its reported driving scores are not lap time measurements.

## Literal racing example with an important reward limitation

[roboserg/PyRacer-env-VLA](https://github.com/roboserg/PyRacer-env-VLA) includes supervised fine tuning of SmolVLM on a toy racing environment and a purported GRPO script.

In the inspected [train_rl.py](https://github.com/roboserg/PyRacer-env-VLA/blob/main/scripts/train_rl.py), candidate rewards are computed using the same observation before an action is executed. The reward function accepts an action argument but does not use it. Consequently, candidates receive the same driving component of reward, with differences attributable to the text verbosity penalty. Only afterward is the selected action executed. This code does not establish learning which candidate improves subsequent driving.

Use its supervised action representation as a reference, but do not treat this RL implementation as validated evidence for racing improvement. No third party code or weights are bundled here.

## Implications for Thunderhill

No verified public example was found of a 135M, 256M, or 500M total language model trained to optimize racing lap times directly. That is a search finding, not proof that none exists.

RobotxR1 is the most concrete starting reference. A smaller language model with structured telemetry is a reasonable experiment, but its success remains unproven. Camera input would add a vision model and its associated compute cost.

Our Godot environment must advance simulation explicitly after each action, reward the resulting trajectory, and preserve full reset state for fair candidate comparisons. Evaluation must measure legal completed laps and lap times, separately from format compliance. Harbor integration, motorcycle control, physical validation, and checkpoint replay recording remain implementation work.
