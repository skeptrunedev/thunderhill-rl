# Reference evaluation

These are candidates to test, not validated runtime selections. Source review is separate from execution and physical validation.

## Detailed dynamics

[TUM motorcycle model](https://github.com/TUMFTM/motorcycle_model) provides an MBSim multibody motorcycle, parameter data, steering and speed controllers, and an FMU export path. Evaluate tire contact, steering torque, lean, suspension, powertrain representation, reset behavior, and simulation throughput. The newer model shown in the README is not established as the public version. Do not infer public source capabilities from that video.

## Racing baseline

[Nonplanar Vehicle Control](https://github.com/thomasfork/Nonplanar-Vehicle-Control) provides Python and CasADi motorcycle dynamics, tire forces, nonplanar roads, and minimum time racelines. Its [paper](https://arxiv.org/abs/2406.01726) is the specification reference. Suspension and powertrain simplifications prevent treating it as the complete realistic game model. Evaluate numerical behavior over the valid action domain and repeatable initial conditions before using it for training or comparisons. An optimized raceline is a baseline within its model assumptions, not proof of a globally optimal real motorcycle lap.

## Unity integration

[ML Motorcycles](https://github.com/mbaske/ml-motorcycles) demonstrates motorcycle observations, controls, and rewards using Unity ML Agents. Read the existing agent and reset implementation before designing our integration. Its older dependencies and third party assets require separate review.

[Sportbike Sim](https://github.com/strong-ery/sportbike-sim) is a presentation and audio reference only at this stage. Source in `MotorcycleBalanceController.cs` adds steering assistance forces and cornering downforce. The README describes assisted handling and incomplete falling behavior. These must not silently become physical assumptions in our simulator.

## Language model training

[MindDrive](https://github.com/xiaomi-mlab/MindDrive) releases online PPO training of a language decision model through CARLA. A separate action expert translates language decisions into trajectories. It establishes a relevant training reference, not motorcycle racing capability.

[TRL Harbor integration](https://github.com/huggingface/trl/blob/main/docs/source/harbor.md) documents an external agent harness where the trainer owns generation and retains token probabilities. Implement the game interface within this pattern. Verify image observation support with the chosen model and integration; do not assume text tool output automatically supplies vision inputs. Preserve the LLM's control responsibility and document any rider assistance.

## Track evidence

* [Speed Secrets map](https://speedsecrets.com/wp-content/uploads/2018/12/Thunderhill-East.pdf): visually inspected. Useful for layout and turn naming. It shows alternative geometry around Turn 5 and does not supply measured elevation or camber.
* [User POV](https://www.youtube.com/watch?v=yVjzZqYbKuM): title verified as First Laps on Thunderhill East's New Surface (POV), by Ken Moto. Full visual lap inspection remains pending.
* [Aprilia RS660 lap](https://www.youtube.com/watch?v=LjmP603avMQ): title verified as Thunderhill East Repave Aprilia RS660 2:03 Lap. Full visual inspection remains pending.
* [BMW S1000RR footage](https://www.youtube.com/watch?v=JJHtjdlzL2Q): title verified as BMW S1000RR at Thunderhill East, First Weekend on New Pavement. Full visual inspection remains pending.
* [Official reconstruction report](https://www.thunderhill.com/news-from-the-hill-1/inside-the-thunderhill-repave-the-engineering-behind-the-new-surface): preserves overall grade and camber while changing curbing and usable width around Turns 3 to 4, 6 to 7, and 10 to 11.
* [USGS elevation access](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download): candidate terrain source. Local coverage, acquisition date, and resolution have not yet been established.

Footage can establish visual correspondence. It cannot by itself establish tire parameters or precise track banking. Confirm the intended Turn 5 route and motorcycle before finalizing geometry or dynamics parameters.
