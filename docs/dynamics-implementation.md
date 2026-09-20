# Initial motorcycle dynamics implementation

The initial Godot model is a reduced order, explicitly stepped prototype for testing controls, scenes, replay and the RL interface. It is not validated hyperrealistic handling. The dynamics version is `reduced-order-rider-v1`.

`godot/scripts/motorcycle.gd` defines `MotorcycleSim`, a `RefCounted` object with no scene processing or wall clock access. Call `reset(position, heading, initial_speed)` and then `step(dt, controls, road_sample)`. Rendering and reading `telemetry()` do not advance it. The caller owns the fixed timestep, at most 0.02 seconds, and road sampling. Ground position is the contact reference. Positive heading turns right, with forward vector `(sin(heading), 0, -cos(heading))`. Positive lean is right. Velocities are metres per second and angles are radians.

## Inputs and assists

Throttle and separate front and rear brakes range from zero to one. Steering ranges from negative one to positive one. Shift is an integer event, negative one, zero or positive one. Release a shift back to zero before repeating it. Invalid numeric input, unknown control names and invalid surface samples return an error and preserve simulation state. The service must separately reject unknown protocol fields, stale actions and retries according to the game contract.

The default `assist_enabled` setting enables an explicit virtual rider. Steering input requests a lean target, and a proportional and damping controller calculates steering to achieve it through the roll equation. The controller initially countersteers when initiating a lean. At low speed, an explicit balancing aid holds the bike upright. With assistance disabled, the command requests steering angle directly, and the natural roll instability remains. These are distinct control experiments and the telemetry labels them.

Automatic shifting is enabled by default and can be disabled with `auto_shift`. An automatic launch clutch is always present in this prototype. Tire forces are limited ideally at the grip envelope, approximating protective ABS and traction control behavior. This does not emulate Ducati's proprietary electronics. The force limiter remains active in direct steering mode and is always labeled in telemetry.

## Forces and state

Engine torque passes through the selected primary, gear and final ratios to the rear wheel. A broad estimated torque curve respects the published US peak torque and power. Shift interruption, drag, rolling resistance, engine braking and road grade affect speed. Separate brake requests act on their corresponding axle. Estimated mass, center of gravity and previous acceleration determine longitudinal load transfer, with drive unloading the front axle and braking loading it.

Each tire's longitudinal force consumes part of a circular friction budget. The remaining front and rear budgets bound total lateral force. Steering requests lateral force using wheelbase and speed. The roll equation includes gravity and lateral acceleration; exceeding the ground contact lean threshold marks a fall. The fallen state slides to rest under a labeled estimated deceleration. The game server should stop RL episodes at the fall event, although human visualization may continue the slide.

Leaving the road lowers grip and increases rolling resistance. It does not teleport or steer the motorcycle back onto the course. Track boundary and lap legality decisions belong to the episode and track logic.

Important simplifications include no separate lateral velocity or wheel slip state, no tire relaxation or thermal model, no suspension, pitch, airborne motion, wheelies or crash collision geometry. The tire model is a force budget, not Pacejka. Road normal affects grade and available normal force, but banked roll dynamics are not modeled. Axle loads use the previous acceleration, which is an explicit numerical approximation rather than a coupled pitch solution. Changes in surface and sharp throttle or brake commands require timestep sensitivity testing.

## Provenance and numerical reference

The published dimensions, mass without fuel, and gear ratios are recorded in [the manufacturer reference](motorcycle-reference.md), sourced from the [Ducati US page](https://www.ducati.com/us/en/bikes/streetfighter/streetfighter-v4). Rider and fuel masses, torque shape, loaded rolling radius, drag area, center of gravity, grip and controller settings are configurable estimates. They must not be presented as measured values of the user's motorcycle.

The independent research reference is [Nonplanar Vehicle Control](https://github.com/thomasfork/Nonplanar-Vehicle-Control), particularly its [motorcycle model](https://github.com/thomasfork/Nonplanar-Vehicle-Control/blob/main/src/vehicle_3d/models/motorcycle_model.py), [tire model](https://github.com/thomasfork/Nonplanar-Vehicle-Control/blob/main/src/vehicle_3d/models/motorcycle_tire_model.py) and explicit dynamics stepping. Its published model and code were inspected and executed during research. The execution audit found invalid tire force domains under unrestricted actions and an asymmetric initial load issue. This implementation therefore treats grip capacity and failure reporting explicitly, but does not reproduce that research model or claim its validation. No upstream source or mesh is copied here.

## Acceptance boundary

Deterministic reset, numerical finiteness, acceleration and braking direction, correctly signed load transfer, countersteering and combined grip constraints can be tested now. Calibration against measured acceleration, braking, lean response, suspension and tire data remains outstanding. Successful laps with this prototype demonstrate the game loop and assisted control task, not physical fidelity or optimal real world racing.
