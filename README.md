# Thunderhill motorcycle RL

A Godot motorcycle racing environment for gameplay reinforcement learning on Thunderhill East. The track uses documented historical geometry with appearance informed by recent footage. No paid engine license is required.

## Acceptance criteria

1. Motorcycle dynamics are evaluated against published models and measured behavior. Steering, tire forces, lean, braking, suspension, and rider movement must have explicit physical assumptions.
2. The track uses documented geometry, elevation, camber, and current surface references. Unknown measurements remain marked as unknown.
3. NVIDIA AlpaGym and Cosmos RL train Alpamayo's trajectory expert from simulator rewards. The vision language backbone remains frozen. No supervised imitation is used.
4. Simulation time advances explicitly in fixed steps. Model inference time does not advance the world.
5. Evaluation compares legal laps, completion rate, crashes, and lap times under identical conditions.
6. Model checkpoints and state trajectories support recordings of training progress from consistent camera views.

Current status: the Godot game supports human controls, agent stepping, camera observations, telemetry recording, and state replay. The custom Alpamayo training loop has been replaced by an adapter to NVIDIA AlpaGym and Cosmos RL. Full GPU training through the replacement stack remains unverified. See [training instructions](training/README.md).

The motorcycle references the 2026 Ducati Streetfighter V4S. The [reference readiness report](docs/reference-readiness.md) records inspected footage, acquired terrain and materials, the M3 Pro Mac target, and remaining realism gaps.

The [dimension register](docs/dimension-register.md) records measured aerial widths, raw lidar grades and cross slopes, bike dimensions, and the existing simulator geometry audit.

See [reference evaluation](docs/reference-evaluation.md) for dynamics and simulator source references, and [playable status](docs/playable-status.md) for game controls and verification commands.

The [RL game design contract](docs/rl-game-contract.md) defines the information the simulator and trainer must capture, action and reward semantics, isolated rollouts, and checkpoint replay requirements.

## License

Original project code and documentation use the [MIT license](LICENSE). Referenced projects, models, footage, maps, and assets retain their own licenses. The measurement databases explicitly marked as derived from OpenStreetMap use ODbL 1.0, with attribution to OpenStreetMap contributors, rather than MIT. Original USGS and USDA geodata is public domain. Third party game assets and video frames are not bundled here. Godot is [MIT licensed](https://godotengine.org/license/).

See [experiment graphs](docs/experiment-tracking.md) for W&B tracking in the current worker stack.
