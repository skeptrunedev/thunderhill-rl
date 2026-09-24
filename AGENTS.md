# Training instructions

Never perform supervised imitation for this project. This includes supervised fine tuning on driving demonstrations, teacher actions, scripted controller actions, recorded action labels, or model generated demonstrations. Do not use it as a bootstrap, recovery step, prerequisite, or substitute for reinforcement learning.

Train through reinforcement learning from the model's own gameplay. The model must observe the simulator, issue native tool calls to control the motorcycle, receive rewards from the resulting simulator transitions, and update from those collected rollouts. Failed attempts are valid learning experience; completing a lap is not a prerequisite for starting RL.

Use the native tool calling format of the selected model. Preserve recordings of every rollout and track generations, rollout identity, rewards, driving outcomes, and actual optimizer updates. Distinguish tool syntax validity from driving ability, and do not claim improvement without measured gameplay evidence.

Existing supervised training scripts and datasets are historical artifacts, not authorization to run them. Do not launch a pipeline that invokes supervised imitation, even if it later includes an RL stage. Only an explicit future user instruction changing this rule can authorize supervised imitation.

# Verification preference

Do not add standalone unit tests. Verify changes through the running game, actual simulator and recording workflows, and real training checks when suitable hardware is available. Preserve the distinction between synthetic inference fixtures and real model training.
