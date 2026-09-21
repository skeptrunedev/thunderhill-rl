# Selected motorcycle reference

The user selected the 2026 Ducati Streetfighter V4 S. Reference checked 19 September 2026. This document records a manufacturer specification baseline, not a validated motorcycle dynamics model.

## Source and generation boundary

The [current Ducati US model page](https://www.ducati.com/us/en/bikes/streetfighter/streetfighter-v4) still has MY25 in its title. Its S column describes the current generation with a symmetrical swingarm, Hypure brakes, and Smart EC 3.0 suspension. Use that column, not the base V4, SP2, or the separate MY24 page. Exact model year confirmation through the 2026 manual remains outstanding. The official [owner manual selector](https://www.ducati.com/ww/en/service-maintenance/owner-manuals) was inspected, but its year and model selection did not expose the requested PDF through the available text fetch. No manual has been represented as read.

## Verified current S baseline

Values below come from Ducati's current US S specification table.

| Parameter | Published value |
| --- | --- |
| Mass | 189 kg, wet without fuel |
| Tank capacity | 16 litres |
| Wheelbase | 1,496 mm |
| Rake / trail | 24.5 degrees / 99 mm |
| Seat height | 845 mm |
| Engine | 1,103 cc V4, 90 degrees, counterrotating crank |
| Bore / stroke / compression | 81 mm / 53.5 mm / 14:1 |
| Front / rear tires | Pirelli Diablo Rosso IV Corsa, 120/70 ZR17 / 200/60 ZR17 |
| Front / rear rims | Forged aluminum, 3.50 × 17 / 6.00 × 17 inches |
| Front suspension | Öhlins NIX30 SV, 43 mm fork, Smart EC 3.0 |
| Rear suspension | Öhlins TTX36 SV, Smart EC 3.0 |
| Front / rear wheel travel | 125 / 130 mm |
| Front brakes | Two 330 mm discs, Brembo Hypure four piston calipers |
| Rear brake | 245 mm disc, two piston caliper |
| Brake electronics | Race eCBS |
| Primary ratio | 1.80:1 |
| Gear ratios, first through sixth | 38/14, 36/17, 33/19, 32/21, 30/22, 30/24 |
| Final drive | 520 chain, 15 tooth front / 42 tooth rear |
| Transmission | Six gears, DQS 2.0, wet slipper clutch |

The [international Ducati specification](https://www.ducati.com/ww/en/bikes/streetfighter/streetfighter-v4) explicitly separates engine ratings by market:

| Market | Peak power | Peak torque |
| --- | --- | --- |
| USA / Canada / Mexico | 150.7 kW (205 hp) at 12,650 rpm | 119.7 Nm at 11,500 rpm |
| Other listed specification | 157.4 kW (214 hp) at 13,500 rpm | 120.0 Nm at 11,250 rpm |

For a Thunderhill baseline, use the US specification as a provisional market assumption. The user's actual exhaust, ECU tune, tires, gearing, and suspension settings have not been identified. Do not silently substitute the international rating or an accessory exhaust rating.

## Modeling implications and unknowns

The published mass excludes fuel. Add the selected fuel load and rider separately; neither is included in 189 kg. Do not call that figure dry mass. Tank capacity is not the initial fuel load. Tire size markings are not measured loaded rolling radii.

The brochure values do not establish the following inputs:

1. Bike and rider center of gravity, inertia tensors, axle load split, and rider motion limits.
2. Tire force coefficients, combined slip behavior, camber thrust, temperature response, carcass stiffness, pressure, or pavement grip.
3. Spring rates, damping curves, preload, sag, suspension linkage geometry, and electronic suspension control laws.
4. Full torque versus rpm and throttle map, engine braking map, drivetrain losses, rotating inertia, clutch response, and shift timing.
5. Aerodynamic drag, lift, and downforce curves as functions of speed, lean, yaw, and rider posture.
6. Hydraulic brake gain, pad friction, thermal response, or proprietary traction control, wheelie control, ABS, and eCBS logic.

Keep estimated parameters explicitly labeled, configurable, and versioned. Manufacturer dimensions constrain the visual model, but do not validate handling. Stock tires and stock electronics are a baseline configuration to model deliberately, not permission to invent their behavior. A credible handling claim requires measured acceleration, braking, turn response, and suspension behavior under known rider, tire, fuel, and assist settings.

The next data acquisition priority is the exact 2026 US owner manual and the actual reference bike setup. Preserve this distinction while beginning original geometry and a tunable physics prototype. No Ducati imagery, manual pages, or third party motorcycle mesh is bundled with this document.


## Cockpit reference consistency check, 20 September 2026

The official current model electronics page specifies a 6.9 inch diagonal,
8:3 aspect ratio, 1280 by 480 TFT:
https://www.ducati.com/us/en/bikes/streetfighter/streetfighter-v4/electronics
This was fetched directly during the cockpit audit. The game's current visible
face is 0.177 by 0.097 metres with a 1024 by 560 render texture; those proportions
are not the manufacturer's current display specification. The supplied frame
00:10 appears to show a less wide display, but perspective alone cannot establish
its exact model generation. Do not describe the existing cockpit as an exact
2026 reconstruction or silently combine a current specification with an older
visual reference. User preference between video appearance and the current
manufacturer model was requested during this audit.

`godot/tools/preview_cockpit.gd` records the actual production onboard pose and
three alternative camera estimates at fixed 1280 by 720 resolution. Production
uses the actual camera transform from `_update_visual`; alternatives use stated
bike local anchors, downward pitch and field of view. Geometry and lighting are
fixed. Captures and source hashes are in
`artifacts/cockpit-framing-verified/study.json`. The earlier
`artifacts/lighting-production.png` is 1152 by 720, so direct pixel comparisons
against the 1280 by 720 footage were not matched in aspect ratio.

The alternatives do not establish a recovered lens calibration. They confirm
that framing can change the apparent tank and cockpit coverage substantially.
The reference supports a missing dark headstock and ignition assembly, but one
obscured onboard view is insufficient evidence for a particular tank recess.
No tank reshaping or production camera change was adopted from this study.


## Original headstock hardware study

The cockpit now includes an original hexagonal steering stem fastener and a
separate cylindrical ignition housing with a metallic rim, inserted key,
mounting plate and short support. Their dimensions and chassis attachment are
visual estimates, not an OEM mounting reconstruction. The requested official
Ducati installation PDF could not be downloaded (HTTP 403); no mounting claim
is based on that unread document.

The stem fastener follows the visual steering assembly. The ignition stays on
the chassis. Clearance was checked against the modeled yoke and headlight
through the existing steering range of plus or minus 0.5 radians. The final
housing is centered at bike local (0, 0.925, -0.570) metres. The plate is 44 by
10 by 16 mm with a 3 mm bevel; its narrower footprint resolves a headlight
intersection found during the full steering sweep. These are authoring values,
not measured Ducati dimensions. Tank and production camera geometry are unchanged.

The inspection tool accepts `--steering` in radians, validates against the
simulation limit and records the visual angle. Earlier capture directories
`cockpit-headstock`, `cockpit-headstock-clearance` and `headstock-final-left`
contain intermediate placements. The final bracket capture is
`artifacts/headstock-bracket-verified/production.png`; the neutral view in
`artifacts/headstock-verified/production.png` predates only the bracket narrowing.
The new hardware adds recognizable detail but is not positioned like the
obscured ignition area in the video. It does not resolve the tank silhouette,
display generation mismatch or the broader cockpit realism gap. The existing
front assembly steering pivot is also a simplified visual mechanism.
