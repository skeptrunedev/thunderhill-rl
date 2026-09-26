# Training instructions

The primary approach is reinforcement learning from the agent's own gameplay: the policy observes the simulator, controls the motorcycle, receives rewards from the resulting simulator transitions, and updates from the collected rollouts. Failed attempts are valid learning experience; completing a lap is not a prerequisite for starting RL.

Offline and supervised training are allowed, for example a decision transformer trained on the SAC agent's recorded rollouts.

Preserve recordings of every rollout and track generations, rollout identity, rewards, driving outcomes, and actual optimizer updates. Do not claim improvement without measured gameplay evidence.

# Verification preference

Do not add standalone unit tests. Verify changes through the running game, actual simulator and recording workflows, and real training checks when suitable hardware is available. Preserve the distinction between synthetic inference fixtures and real model training.
