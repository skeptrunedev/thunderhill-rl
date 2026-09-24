# Alpamayo gameplay RL validation

The first experiment is one bounded generation using the published Alpamayo 1.5
driving checkpoint. It must learn from its own Thunderhill simulator outcomes.
No supervised imitation, teacher actions, demonstration dataset, or distillation
is part of this experiment.

The policy predicts a trajectory from camera frames and motion history. A fixed
motorcycle controller executes that trajectory through the simulator controls.
These controller commands are not native language model tool calls. The policy
gradient must use the actual sampled trajectory probability, never the text
serialization of the controller commands.

The integration reference is NVIDIA AlpaGym's continuous action expert replay:
https://github.com/NVlabs/alpagym

The initial cost boundary is one RTX PRO 6000, at most 25 minutes of child GPU
execution, no automatic retries, and one generation. Start with two training
rollouts and matching evaluations before and after the update, each capped at
12 simulated seconds. Record every attempt and queue videos for local rendering.
Download model weights using CPU resources before allocating the GPU.

Completion requires audited gameplay rewards, an actual nonzero optimizer update,
a saved reloadable checkpoint, evaluation of that checkpoint, complete video
coverage, and confirmation that the cloud job has terminated. One generation is
a pipeline validation, not statistical evidence of improved lap performance.

Local validation now covers real rendered camera capture, trajectory execution,
reward auditing, recording queues and persisted replay artifacts. CPU tests of
NVIDIA's stochastic sampler verify replay density and policy gradient arithmetic
using a tiny test head, not a driving checkpoint.

The actual processor test remains gated by access to `nvidia/Cosmos-Reason2-8B`.
The full Alpamayo model has not completed a gameplay or optimizer validation.
Neither local integration tests nor the CPU arithmetic tests establish that the
complete model trains successfully on the planned GPU.
