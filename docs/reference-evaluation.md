# Reference evaluation

These are candidates to test, not validated runtime selections. Source review is separate from execution and physical validation.

## Detailed dynamics

[TUM motorcycle model](https://github.com/TUMFTM/motorcycle_model) provides an MBSim multibody motorcycle, parameter data, steering and speed controllers, and an FMU export path. Evaluate tire contact, steering torque, lean, suspension, powertrain representation, reset behavior, and simulation throughput. The newer model shown in the README is not established as the public version. Do not infer public source capabilities from that video.

## Racing baseline

[Nonplanar Vehicle Control](https://github.com/thomasfork/Nonplanar-Vehicle-Control) provides Python and CasADi motorcycle dynamics, tire forces, nonplanar roads, and minimum time racelines. Its [paper](https://arxiv.org/abs/2406.01726) is the specification reference. Suspension and powertrain simplifications prevent treating it as the complete realistic game model. Evaluate numerical behavior over the valid action domain and repeatable initial conditions before using it for training or comparisons. An optimized raceline is a baseline within its model assumptions, not proof of a globally optimal real motorcycle lap.

## Engine integration

Godot is the selected engine because the project must not require buying an engine license. [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) is an integration reference for observations, actions, and training transport. Its conventional policy backends do not replace the required LLM training integration. [Godot Movie Maker](https://docs.godotengine.org/en/stable/tutorials/animation/creating_movies.html) is a candidate for offline checkpoint footage. Motorcycle dynamics remain subject to the research evaluation above.

[ML Motorcycles](https://github.com/mbaske/ml-motorcycles) demonstrates motorcycle observations, controls, and rewards using Unity ML Agents. Read the existing agent and reset implementation before designing our integration. Its older dependencies and third party assets require separate review.

[Sportbike Sim](https://github.com/strong-ery/sportbike-sim) is a presentation and audio reference only at this stage. Source in `MotorcycleBalanceController.cs` adds steering assistance forces and cornering downforce. The README describes assisted handling and incomplete falling behavior. These must not silently become physical assumptions in our simulator.

## Current training integration

[NVIDIA AlpaGym](https://github.com/NVlabs/alpagym/tree/972d160eed0e23d388497851504a3a233fec5879) supplies the native Alpamayo and Cosmos RL implementation used by this project. Our Godot adapter replaces the simulator boundary. The trajectory expert is trained from gameplay rewards while the vision language backbone remains frozen. This is not the earlier language tool calling approach.

See [the current training instructions](../training/README.md) for setup, actual verification scope and simulator differences. Native model and optimizer code does not establish parity of motorcycle dynamics, camera appearance, reward geometry or achieved driving behavior.

## Track evidence

* [Speed Secrets map](https://speedsecrets.com/wp-content/uploads/2018/12/Thunderhill-East.pdf): visually inspected. Useful for layout and turn naming. It shows alternative geometry around Turn 5 and does not supply measured elevation or camber.
* [User POV](https://www.youtube.com/watch?v=yVjzZqYbKuM): Ken Moto's reference footage. Inspected portions and recovered video provenance are documented in [reference readiness](reference-readiness.md) and [camera registration](reference-camera-registration.md).
* [Aprilia RS660 lap](https://www.youtube.com/watch?v=LjmP603avMQ): title verified as Thunderhill East Repave Aprilia RS660 2:03 Lap. Full visual inspection remains pending.
* [BMW S1000RR footage](https://www.youtube.com/watch?v=JJHtjdlzL2Q): title verified as BMW S1000RR at Thunderhill East, First Weekend on New Pavement. Full visual inspection remains pending.
* [Official reconstruction report](https://www.thunderhill.com/news-from-the-hill-1/inside-the-thunderhill-repave-the-engineering-behind-the-new-surface): preserves overall grade and camber while changing curbing and usable width around Turns 3 to 4, 6 to 7, and 10 to 11.
* [USGS elevation access](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download): acquired terrain and lidar provenance is recorded in [geometry sources](geometry-sources.md) and [lidar measurements](lidar-measurements.md).

Footage can establish visual correspondence. It cannot by itself establish tire parameters or precise track banking. The selected route uses the Cyclone and the selected motorcycle is the Streetfighter V4 S. Published dimensions do not establish calibrated dynamics.
