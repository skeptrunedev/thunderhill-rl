# Initial motorcycle dynamics implementation

The initial Godot model is a reduced order, explicitly stepped prototype for testing controls, scenes, replay and the RL interface. It is not validated hyperrealistic handling. The dynamics version is `reduced-order-surface-grip-v6`.

`godot/scripts/motorcycle.gd` defines `MotorcycleSim`, a `RefCounted` object with no scene processing or wall clock access. Call `reset(position, heading, initial_speed)` and then `step(dt, controls, road_sample)`. Rendering and reading `telemetry()` do not advance it. The caller owns the fixed timestep, at most 0.02 seconds, and road sampling. Ground position is the contact reference. Positive heading turns right, with forward vector `(sin(heading), 0, -cos(heading))`. Positive lean is right. Velocities are metres per second and angles are radians.

## Inputs and assists

Throttle and separate front and rear brakes range from zero to one. Steering ranges from negative one to positive one. Shift is an integer event, negative one, zero or positive one. Release a shift back to zero before repeating it. Invalid numeric input, unknown control names and invalid surface samples return an error and preserve simulation state. The service must separately reject unknown protocol fields, stale actions and retries according to the game contract.

The default `assist_enabled` setting enables an explicit virtual rider. Steering input requests a lean target, and a proportional and damping controller calculates steering to achieve it through the roll equation. The controller initially countersteers when initiating a lean. At low speed, an explicit balancing aid holds the bike upright. With assistance disabled, the command requests steering angle directly, and the natural roll instability remains. These are distinct control experiments and the telemetry labels them.

Automatic shifting is enabled by default and can be disabled with `auto_shift`. An automatic launch clutch is always present in this prototype. Tire forces are limited ideally at the grip envelope, approximating protective ABS and traction control behavior. This does not emulate Ducati's proprietary electronics. The force limiter remains active in direct steering mode and is always labeled in telemetry.

## Forces and state

Engine torque passes through the selected primary, gear and final ratios to the rear wheel. A broad estimated torque curve respects the published US peak torque and power. Shift interruption, drag, rolling resistance, engine braking and road grade affect speed. Separate brake requests act on their corresponding axle. Estimated mass, center of gravity and current tire forces determine a simultaneous longitudinal force and axle load balance, with drive unloading the front axle and braking loading it.

Each tire's longitudinal force consumes part of a circular friction budget. The remaining front and rear budgets bound total lateral force while preserving steady yaw moment balance about the center of gravity. Steering requests lateral force using wheelbase and speed. The roll equation includes gravity and lateral acceleration; exceeding the ground contact lean threshold marks a fall. The fallen state slides to rest under a labeled estimated deceleration. The game server should stop RL episodes at the fall event, although human visualization may continue the slide.

Unpaved ground lowers grip and increases rolling resistance. Exposed curb contact uses separate configurable curb coefficients, independently of track legality. It does not teleport or steer the motorcycle back onto the course. Track boundary and lap legality decisions belong to the episode and track logic.

Important simplifications include no separate lateral velocity or wheel slip state, no tire relaxation or thermal model, no suspension, pitch, airborne motion, wheelies or crash collision geometry. The tire model is a force budget, not Pacejka. Bank and grade gravity are projected into a local road frame, with the contact assumptions explained below. Axle loads satisfy quasistatic pitch balance within the declared contact model. Axle lateral forces now satisfy steady yaw moment balance. There is still no yaw inertia or transient yaw response, and steered front tire forces are not rotated into the road frame; this is a small steering angle force approximation. Changes in surface and sharp throttle or brake commands require timestep sensitivity testing.

## Axle lateral force balance

Version 4 replaces the sum of remaining tire budgets with a steady yaw balance constraint. Let `f` be the static front load fraction and `L` the wheelbase. The distances from the center of gravity to the front and rear contacts are `a=(1-f)*L` and `b=f*L`. Gravity acts through the center of gravity. With zero yaw acceleration, axle forces must satisfy:

```text
a*front_lateral_force - b*rear_lateral_force = 0
front_lateral_force = f*total_lateral_force
rear_lateral_force = (1-f)*total_lateral_force
front_remaining = sqrt((mu*front_load)^2 - front_longitudinal_force^2)
rear_remaining = sqrt((mu*rear_load)^2 - rear_longitudinal_force^2)
total_lateral_limit = min(front_remaining/f, rear_remaining/(1-f))
```

The geometric force share uses the static fraction, not the instantaneous load fraction altered by braking. Requested total lateral force is clamped to this limit, then allocated to each axle. A fully saturated front brake therefore cannot retain steady cornering using only the rear tire. Direct steering mode retains longitudinal priority in the ideal force limiter. Assisted mode explicitly reserves lateral grip as described below.

Telemetry adds `front_lateral_force_n`, `rear_lateral_force_n`, `front_lateral_capacity_n`, `rear_lateral_capacity_n`, `lateral_capacity_n` and `requested_lateral_force_n`. Forces are along road right. Capacities are the remaining lateral budgets after longitudinal demand. These values reset to zero before the first transition. Unsupported telemetry explicitly includes `yaw_inertia` and `steered_tire_force_rotation`. This algebraic balance does not reproduce transient yaw, lateral slip, camber thrust or individual tire slip angles.

`tests/test_lateral_axles.gd` exercises 60 combinations of asymmetric center of gravity locations, front and rear braking, small and saturated lateral demands, and mirrored turns. It independently checks zero yaw moment, Newton lateral force balance, both friction circles, unconstrained demands, saturation of the limiting axle, zero cornering capacity with a saturated brake, and reset and telemetry values. The test uses a lower configured center of gravity to isolate lateral force sharing while both contacts remain supported; it is a synthetic mechanics case, not motorcycle calibration.

## Assisted combined force allocation

Version 5 adds an explicit ideal traction and brake intervention to assisted mode. The version 4 axle correction exposed a real force allocation problem: full throttle consumed the entire rear friction circle, leaving neither axle able to generate the steady balanced lateral force needed on the slightly banked starting straight. The virtual rider could request corrective steering but had no available force. This caused an upright straight launch to fall. The correction reduces longitudinal force rather than inventing lateral grip.

After calculating the rider's steering and lateral force request, the allocator begins with the valid longitudinal contact solution `(Xf, Xr)`. It reserves the lateral request, bounded by the pure lateral limit `mu*N`, and seeks the largest common longitudinal scale `s` between zero and one satisfying:

```text
Nf(s) = f*N - (h/L)*s*(Xf+Xr)
Nr(s) = N - Nf(s)
(s*Xf)^2 + (f*reserved_lateral)^2 <= (mu*Nf(s))^2
(s*Xr)^2 + ((1-f)*reserved_lateral)^2 <= (mu*Nr(s))^2
Nf(s) >= 0, Nr(s) >= 0
```

Each friction constraint is a convex norm bounded by an affine nonnegative load. Scale zero is feasible for the bounded lateral request, so feasible scales form an interval containing zero. A deterministic bisection finds its upper endpoint, with an immediate return when full longitudinal force is feasible. Loads and acceleration are recomputed from the reduced forces in the same tick. The scalar intervention preserves the initially limited front and rear force ratio; it is not an optimal independent axle brake controller. No measured electronic control behavior is claimed.

The same policy applies to breakaway after a stop event. A held stationary reaction is exempt because scaling it would violate grade force balance. Direct mode retains its declared longitudinal priority. Telemetry identifies `combined_force_policy` and `longitudinal_force_scale`; scale one means no reduction by this allocator, not absence of the original tire force limiter. All control actions and applied actuator values remain observable.

`tests/test_combined_assist.gd` reproduces straight full throttle launches on mirrored banks at the recorded starting bank magnitude (approximately 0.018 radians), and on larger 0.08 radian banks. Every tick checks both friction circles, pitch balance and yaw balance. It follows each launch with full front and rear braking. Independent cases check the closed form circular limit with zero center of gravity height and unchanged maximum braking when no lateral force is requested. The synthetic parameters are mechanics tests, not calibration. Actual rendered scene controls and a complete legal QA lap are separate integration checks.

## Longitudinal contact and stationary braking

Version 3 removes the artificial previous acceleration memory from axle loads. With total normal support `N`, configured static front fraction `f`, height `h` and wheelbase `L`, the same tick must satisfy:

```text
front_load = f*N - (h/L)*(front_tire_force + rear_tire_force)
rear_load = N - front_load
```

Each longitudinal tire force is its requested force limited by that tire's available friction. These relationships are piecewise linear in front load. The solver enumerates the saturation breakpoints, solves each interval exactly and accepts a unique solution with nonnegative front and rear loads. It never floors a negative normal force to keep a physically impossible two contact state running.

Aerodynamic drag acts through the center of mass in this reduced model and contributes no pitching moment. Effective rolling resistance is also treated as a translational center of mass force, an explicit simplifying assumption rather than a measured tire deformation or rolling moment model. The contact height and static load split remain configured estimates. Dynamic pitch, suspension response and the complete leaned multibody contact geometry are not inferred from this algebraic model.

At rest, commanded braking is a capacity constraint. A signed common fraction of the front and rear commands is chosen to oppose the actual drive and grade force. This allows a parked motorcycle to remain stationary on either incline direction without reporting fictitious full braking deceleration. If brakes cannot hold, signed motion begins under the remaining force. There is no hidden hill hold. `longitudinal_velocity` is now the signed velocity along heading; `speed` remains its nonnegative magnitude for existing instruments and observations. Forward drive remains forward during rollback, while brake, engine braking and resistance signs oppose travel. This is rollback with an automatic slipping clutch, not a reverse gear model.

Rollback drivetrain behavior remains an approximation: engine speed is computed from wheel speed magnitude, and the engine braking term is a symmetric resisting torque. A real forward gearbox cannot impose that coupling during reverse wheel rotation without clutch slip or disengagement. The prototype does not solve clutch engagement, clutch slip torque, clutch temperature or engine rotational inertia. Signed rollback supports starts on grades, but backward drivetrain behavior is not a validated motorcycle model.

When braking reaches zero velocity within a tick, position is integrated to the exact stop time. The stationary balance then governs the remaining tick, allowing holding or breakaway in either direction. Telemetry acceleration and force describe the end state; they are not necessarily the average acceleration across a tick containing a stop event. Normal forward motion uses the existing fixed step force integration.

If no unique two contact solution exists, the simulator returns `failure_type: unsupported_dynamics` with a reason and unchanged pretransition state. Actuator, drivetrain, position and tick changes are rolled back atomically for that rejected transition. This is an invalid rollout caused by missing physical capability, not a rider crash or rewardable task outcome. No airborne or pitch trajectory is fabricated.

`tests/test_longitudinal.gd` checks independent closed form saturated front and rear braking solutions, zero dependence on previous acceleration, static force balance on flat ground and both incline directions, unbraked rollback, stopping distance from `v²/(2a)`, braking while rolling backwards and atomic rejection of infeasible high grip contact. The original bank and motorcycle regressions also remain applicable.

## Bank and inclined plane mechanics

The forward vector keeps the requested horizontal heading while satisfying the road plane constraint. The road right axis is forward crossed with the upward surface normal. Gravity projects independently along forward, right and the negative normal. This avoids treating downhill acceleration as tire force or consuming tire grip for gravity itself.

For a locally fixed plane, normal support is mass times the gravity normal component. Tire lateral force is mass times the difference between requested lateral acceleration and gravity's road right component. Tire friction limits apply to that tire force, not total acceleration. If friction is zero, the model therefore accelerates down a bank even with straight steering.

`lean` remains the angle away from world upright projected perpendicular to travel. Positive lean points right. `road_bank_rad` measures the surface normal's right tilt from that upright axis, and `lean_relative_road_rad = lean - road_bank_rad`. Collision clearance uses relative lean. Existing camera and mesh code can continue using world lean. The road plane roll equation is:

```text
roll_acceleration = (gravity_normal * sin(relative_lean)
                    + (gravity_right - lateral_acceleration) * cos(relative_lean)) / cg_height
```

The rider controller solves this equation for the required lateral acceleration while retaining the existing world lean target. The stationary roll equilibrium is therefore `tan(relative_lean) = (lateral_acceleration - gravity_right) / gravity_normal`. The controller is not allowed to substitute an arbitrary force to hold its requested lean. Horizontal heading rate includes the plane to horizontal projection, so turning on a grade or bank does not silently reinterpret the heading coordinate.

Telemetry adds `road_bank_rad`, `lean_relative_road_rad`, `gravity_forward_m_s2`, `gravity_right_m_s2` and `gravity_normal_m_s2`. Existing `lateral_acceleration` is now explicitly total acceleration along the road right axis; `lateral_force_n` remains tire force. All additions reset deterministically. Saved old trajectories remain world lean trajectories, but model version must distinguish training comparisons.

The [primary nonplanar motorcycle paper](https://arxiv.org/html/2406.01726v1), particularly its road frame, gravity and force balance sections, motivates separating road and motorcycle frames. The implemented equations above are a simpler locally planar balance, not a reproduction of the paper's multibody model. That distinction matters for curved surfaces: a banked circular bowl, a crest and a compression require normal acceleration from surface curvature. The present normal force excludes that term. It also excludes moving frame rotational inertial terms. These limitations are emitted as `surface_curvature_normal_load` and `surface_frame_rotation_dynamics` in unsupported telemetry, replacing the former blanket bank limitation. No aerodynamic downforce or empirical normal force boost is introduced.

Independent Godot tests in `tests/test_banking.gd` cover mirrored inclined planes, normal support equal to `mg cos(bank)`, straight contour riding with uphill friction, frictionless downslope acceleration equal to `g sin(bank)`, zero roll moment at the analytically derived equilibrium, upright assisted contour riding and preservation of horizontal heading on combined grade and bank.

## Next contact and suspension work

The next physical extension needs a road query at front and rear contact positions, continuous geometric derivatives or an independently checked curvature estimate, and separate body and wheel vertical states. Surface curvature supplies the required normal acceleration. A normal force cannot become tensile: loss of support must transition to an airborne state rather than pinning the motorcycle to terrain. Landing requires contact impulses or a compliant contact model and energy accounting. These transitions need flat drop, crest release and landing tests before they can be accepted.

Suspension additionally requires sprung and unsprung masses, inertias, spring and damper curves, installed geometry, sag and travel stops. Published wheel travel alone does not determine those values. No spring constants or contact damping are invented in this change. Preserve the current explicitly limited contact model until a tested replacement includes those states, contact transitions and reset snapshots.

## Provenance and numerical reference

The published dimensions, mass without fuel, and gear ratios are recorded in [the manufacturer reference](motorcycle-reference.md), sourced from the [Ducati US page](https://www.ducati.com/us/en/bikes/streetfighter/streetfighter-v4). Rider and fuel masses, torque shape, loaded rolling radius, drag area, center of gravity, grip and controller settings are configurable estimates. They must not be presented as measured values of the user's motorcycle.

The independent research reference is [Nonplanar Vehicle Control](https://github.com/thomasfork/Nonplanar-Vehicle-Control), particularly its [motorcycle model](https://github.com/thomasfork/Nonplanar-Vehicle-Control/blob/main/src/vehicle_3d/models/motorcycle_model.py), [tire model](https://github.com/thomasfork/Nonplanar-Vehicle-Control/blob/main/src/vehicle_3d/models/motorcycle_tire_model.py) and explicit dynamics stepping. Its published model and code were inspected and executed during research. The execution audit found invalid tire force domains under unrestricted actions and an asymmetric initial load issue. This implementation therefore treats grip capacity and failure reporting explicitly, but does not reproduce that research model or claim its validation. No upstream source or mesh is copied here.

## Acceptance boundary

Deterministic reset, numerical finiteness, acceleration and braking direction, correctly signed load transfer, countersteering and combined grip constraints can be tested now. Calibration against measured acceleration, braking, lean response, suspension and tire data remains outstanding. Successful laps with this prototype demonstrate the game loop and assisted control task, not physical fidelity or optimal real world racing.

## Surface grip selection

Version 6 accepts an optional boolean `on_curb` in the road sample. Exposed
curbs use `curb_friction` and `curb_rolling_coefficient`; otherwise `on_track`
selects the existing asphalt or offroad parameters. Omitted curb flags preserve
legacy callers. Terrain covering a curb has no exposed curb flag. Invalid flags
and nonnumeric, nonfinite or negative selected coefficients reject the transition
without mutating state.

The default curb coefficients reuse the existing dry pavement estimates (1.10
friction and 0.015 rolling resistance). These are placeholders, not measured
paint, concrete or Thunderhill values. Wetness and paint specific grip are not
modeled. This removes the accidental assignment of dirt coefficients to exposed
curbs without claiming calibrated tire behavior.

`state.surface_material`, `state.surface_friction` and
`state.surface_rolling_coefficient` describe the contact selected at the start of
the last accepted step. Before any step they are `unknown`, zero and zero.
`track.on_curb` describes the sampled position after the step, so a transition
crossing a boundary may legitimately have different current contact and force
material. Both are recorded. Crashed sliding still uses the existing fixed slide
deceleration rather than these tire coefficients. Track legality and rewards
remain independent.
The current model still samples one ground point for both tires; mixed axle
surfaces require separate wheel contact states in the future contact model.
