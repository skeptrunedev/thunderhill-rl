# Selected car reference

The car is a Mazda MX-5 Cup (ND2, the Global MX-5 Cup car built by Mazda and
Long Road Racing), a spec racer that runs at Thunderhill. Reference checked 27
September 2026. This document records a published specification baseline and
the estimates layered on it. It is not a validated vehicle dynamics model.
Select it with `--vehicle=car` (game), `--vehicle car` (training/sac_async.py,
tools/drive_lap.py). The motorcycle stays the default.

## Sources

- [Mazda MX-5 (ND), Wikipedia](https://en.wikipedia.org/wiki/Mazda_MX-5_(ND)): dimensions, 2019+ 2.0 L engine rating, curb weight.
- [Global MX-5 Cup, Wikipedia](https://en.wikipedia.org/wiki/Global_MX-5_Cup): ND2 Cup engine (181 hp, 7,500 rpm limit), SADEV sequential gearbox, Bosch ECU, BFGoodrich tires.
- [Roadster.Blog, MX-5 ND Skyactiv transmission](https://www.roadster.blog/2015/03/mx-5-skyactiv-transmission.html): six speed manual ratios and final drive.
- [Tom Long Racing, the new MX-5 Cup gearbox](https://tomlongracing.com/2020/07/07/getting-technical-with-the-new-mazda-mx-5-cup-gearbox/): sequential shifts under 100 ms.
- A search result snippet for the iRacing MX-5 Cup wiki (the page itself returned HTTP 402) gave the ND1 Cup minimum weight 1,122.5 kg (2,475 lb). Whether that includes the driver was not confirmed.

## Published values used

| Parameter | Value | Source |
| --- | --- | --- |
| Wheelbase | 2,310 mm | Wikipedia ND |
| Length / width / height | 3,915 / 1,735 / 1,235 mm | Wikipedia ND (visual only) |
| Engine | 2.0 L Skyactiv-G I4, 135 kW (181 hp) at 7,000 rpm, 205 N m at 4,000 rpm | Wikipedia ND (2019+) and Global MX-5 Cup |
| Rev limit | 7,500 rpm | Global MX-5 Cup |
| Gear ratios | 5.087, 2.991, 2.035, 1.594, 1.286, 1.000 | Roadster.Blog, ND six speed manual |
| Final drive | 2.866 | Roadster.Blog, US and UK manual |
| Shift time | 0.08 s (under 100 ms) | Tom Long Racing |

The Cup car's SADEV sequential gearbox ratios and final drive were not found.
The model uses the OEM manual ratios above, which is what the ND1 Cup car ran.
That is a documented substitution, not a claim about the ND2 SADEV box.

## Estimates (not sourced; tune freely, bump MODEL_VERSION)

| Parameter | Value | Basis |
| --- | --- | --- |
| Mass | 1,050 kg car + 80 kg driver + 25 kg fuel = 1,155 kg | Near the 1,122.5 kg Cup minimum plus fuel |
| Front / rear track | 1,495 / 1,505 mm | Commonly quoted ND figures, not confirmed from a primary source |
| Static front weight | 50% | Mazda's 50:50 marketing claim, unverified for the Cup car |
| CG height | 0.44 m | Lowered sports car estimate |
| Yaw inertia | 1,350 kg m^2 | About m (0.5 L)^2 |
| Tire rolling radius | 0.305 m | 610 mm overall diameter race tire |
| Torque curve | Piecewise linear, 130 N m at 1,000 rpm, peak 205 N m at 4,000 rpm, 184 N m at 7,000 rpm (135 kW), 168 N m at 7,500 rpm | Fitted through the two published points; not a dyno map |
| Drivetrain efficiency | 0.88 | Typical RWD manual |
| Rotating mass factor | 1 + 0.04 + 0.0015 (gear x final)^2 (1.36 in first, 1.05 in sixth) | Typical wheel and engine inertia |
| Drag area CdA | 0.78 m^2 | Cd about 0.40 on about 1.9 m^2 with cage and hard top |
| Tire peak friction, asphalt | 1.30 | R-compound race tire; curb 1.15, off track 0.48 (the motorcycle's off track value, which the reward's run-off cost assumes) |
| Load sensitivity | friction x (1 - 0.12 (Fz / 2,800 N - 1)) | Typical |
| Magic Formula shape | B 14 front, 20 rear; C 1.45; E 0.1 | Peak slip about 7.6 degrees front and 5.3 degrees rear. The stiffer rear sets a stable understeer balance |
| Roll stiffness front share | 0.55 | Typical |
| Brake capacity / balance | 18 kN at the pedal limit, 70% front | Enough to reach the tire limit; bias from ideal distribution near 1 g |
| ABS | braking uses at most 90% of each wheel's friction limit | The Cup car keeps ABS; modeled as ideal slip control |
| Steering | +/- 0.42 rad road wheel angle, 1.4 rad/s | Rack travel and speed estimates |
| Spin threshold | sideslip beyond 0.8 rad above 4 m/s, or rolling backwards | Termination semantics, not physics |

## Model (godot/scripts/car.gd, MODEL_VERSION mx5-cup-single-track-mf-v1)

- Planar single track (bicycle) kinematics with states forward speed, lateral
  speed and yaw rate, integrated in 8 substeps per 1/120 s tick.
- Four wheel vertical loads with quasi-static longitudinal and lateral load
  transfer from the previous substep's tire forces, split by roll stiffness.
- Per wheel lateral force from a normalized Magic Formula on the axle slip
  angle, scaled by load sensitive friction, combined with the wheel's
  longitudinal force through a friction ellipse. Longitudinal force is the
  commanded drive or brake force clipped at the friction limit (no wheel spin
  state), so full drive can still use up the rear's lateral grip.
- Aero drag, rolling resistance, road grade and cross slope in the road frame,
  with the same surface keys (asphalt, curb, offroad) as the motorcycle.
- Automatic six speed gearbox with shift cut, a slipping launch clutch and
  engine braking; manual shifts through the existing `shift` control.
- The brake pedal is max(front_brake, rear_brake), split by the brake balance.
  `assist_enabled` is accepted and ignored.

Unsupported: suspension dynamics, wheel spin state, tire relaxation length,
temperature, camber, downforce, differential yaw moment, airborne motion and
crash contacts.

### Motorcycle interface substitutes

- `lean` is world roll: road bank plus quasi-static body roll
  (0.0025 rad per m/s^2 of lateral acceleration, outward). `lean_rate` is its
  rate. `lean_relative_road_rad` is the body roll alone.
- The policy observation does not use roll. training/sac_env.py puts sideslip
  and yaw rate in the two lean slots for the car.
- A car cannot fall. A spin sets `crashed` with `crash_reason` "spin". The game
  terminates the episode as a crash and the trainer charges it as the "fall"
  incident, like the motorcycle's lean contact. Leaving the track and obstacle
  contact are unchanged.
- Recordings carry `vehicle`, and `physics_version` is the car MODEL_VERSION.
  Replays and tools/render_run_video.py render the vehicle in the manifest.

## Measured behavior

`godot --headless --path godot --script res://tools/measure_car.gd` (flat
level asphalt, the real CarSim):

| Measure | Model | Published comparison |
| --- | --- | --- |
| Top speed | 219.4 km/h (61 m/s, 5th gear, 7,030 rpm) | MX-5 ND2 about 219 km/h |
| 0 to 100 km/h | 7.0 s | Road car about 6.5 s |
| 100 to 0 km/h | 40.6 m, peak 1.09 g | |
| Steady state lateral | 1.17 g (R40), 1.17 g (R80), 1.16 g (R150), front limited | Target 1.0 to 1.3 g on R-compound tires |

On the track, `tools/drive_lap.py --vehicle car --max-speed 60
--lateral-accel 7 --braking 5 --max-pedal 1.0 --lookahead-base 4
--lookahead-time 0.6` (a privileged path follower, not a racing line)
completes a valid lap in 194.1 s with no off track ticks, 42 m/s maximum,
1.02 g peak lateral and 1.05 g peak braking. With the default 7 m + 1.3 s
lookahead the follower runs wide in a tight corner. Asking it for about 0.7 g of
braking while cornering at 0.75 g spins the car, which is the expected result
of trail braking a car whose rear axle then carries about 36% of the load.
