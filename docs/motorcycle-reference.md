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
