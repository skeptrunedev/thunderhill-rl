# Executed reference checks

Source: [Nonplanar Vehicle Control](https://github.com/thomasfork/Nonplanar-Vehicle-Control), commit `8cb0ee51e2cf5cf7b3abbf21d3bdadc538a6c672`.

Status: synchronous stepping demonstrated; not accepted as production motorcycle physics.

## Environment

The checks ran without a display or renderer using Python 3.11, CasADi 3.6.7, NumPy 2.4.6, and SciPy 1.17.1. Pillow and OpenCV were also required by upstream visualization imports even for this headless check. CasADi 3.8.1 failed at the removed `SX_eye` API. No tracked upstream source was modified.

## Results

| Check | Observed outcome |
| --- | --- |
| Coast at 10 m/s for 100 steps of 0.01 s | 10 m progress, unchanged speed |
| Repeat coast from a fresh state | Identical final state |
| Apply 100 N rear drive for one simulated second | Final speed 10.4166666667 m/s |
| Small steering input | Finite yaw and roll response |
| Zero speed, one step | Finite result |
| Rear force at the global published input maximum | Solver failure before the first completed step |

These are numerical smoke checks, not validation of racing behavior or training throughput.

## Confirmed cause of the extreme input failure

The tire force implementation computes `sqrt(D0**2 - Fx**2)`. The globally allowed action can exceed the individual tire's load dependent friction capacity. The global action box therefore does not define a numerically valid tire domain. An RL policy can encounter this during exploration.

The solution must represent saturation and slip consistently with the physical tire model. Do not conceal it with an arbitrary square root clamp, fabricated state, or retry. Keep this model behind an evaluation boundary until the physical formulation and transition handling are reviewed.

## Additional review findings

`get_empty_state()` divides both tire load expressions by `lr + lr`; the intended wheelbase needs to be checked against `lf + lr`. The symmetric default geometry hides this issue.

Under the positive rear drive check the front normal load increased and the rear normal load decreased. The coordinate conventions and force moment signs still need to be traced. This observation is not yet a diagnosed sign bug.

## Decision

Retain this code as a racing optimization and dynamics reference. It is not currently the trusted physical backend for the Godot game. Continue comparison with the TUM multibody model before committing to the motorcycle runtime.
