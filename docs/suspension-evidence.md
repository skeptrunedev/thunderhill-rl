# Suspension evidence for the selected Streetfighter V4 S

Checked 20 September 2026. Read alongside [motorcycle-reference.md](motorcycle-reference.md). This audit establishes hardware and broad geometry, but does not supply enough measured inputs for a calibrated suspension simulation. No physics constants were changed.

## Applicability

The target is the user's 2026 Streetfighter V4 S. Ducati's current US page still identifies its generation as MY25. Values below are verified for that current S specification, provisionally applicable to the selected bike pending its exact 2026 US manual and component identification. The base V4, previous Smart EC 2.0 generation, Panigale, and aftermarket replacement suspension are separate configurations.

## Published inputs

| Input | Manufacturer evidence | Applicability and modeling limit |
| --- | --- | --- |
| Front fork | Öhlins NIX30 (SV), S EC 3.0, 43 mm, TiN coating | Current S column. Diameter is not spring stiffness or piston area. |
| Rear shock | Öhlins TTX36 (SV), S EC 3.0 | Current S column. Product family does not identify the installed spring or valve calibration. |
| Wheel travel | Front 125 mm, rear 130 mm | Ducati labels these wheel travel. Rear shock stroke is not supplied. |
| Basic geometry | Wheelbase 1,496 mm; rake 24.5 degrees; trail 99 mm | Nominal dimensions do not determine suspension hardpoints or travel dependent geometry. |
| Mass | 189 kg, wet without fuel | Whole motorcycle, not sprung mass; excludes rider and fuel. |

Source: [Ducati US current model, S specification table](https://www.ducati.com/us/en/bikes/streetfighter/streetfighter-v4).

Ducati describes a symmetrical swingarm and rear suspension using a single pull rod, with bearings replacing previous sliding bushings. It describes a shorter shock, but publishes no pivot coordinates or motion ratio curve on this page. [Ducati chassis description](https://www.ducati.com/us/en/bikes/streetfighter/streetfighter-v4/chassis-and-ergonomics).

Ducati's MY25 electronics page describes Smart EC 3.0 adjustment of hydraulic behavior according to riding phase, including lower damping during constant speed travel. Four customizable suspension reference configurations are described. This establishes variable damping behavior, not numeric force versus velocity curves, controller gains or update frequency. This is an India market description, used only for qualitative system operation. [Ducati electronics description](https://www.ducati.com/in/en/bikes/streetfighter/streetfighter-v4-2025/electronics).

## Inputs still missing

No applicable OEM numeric values were verified in the inspected sources for:

1. Front and rear spring rates, spring part numbers, installed preload, free spring lengths or top out springs.
2. Static and rider sag for the actual motorcycle, with rider, fuel and suspension mode recorded.
3. Compression and rebound force curves against shaft speed, displacement, temperature and electronic valve setting.
4. Fork oil level and air spring behavior, seal friction, bump stops and end of travel behavior.
5. Shock length and stroke, linkage hardpoints, wheel to shock motion ratio throughout travel, swingarm pivot and axle geometry.
6. Sprung and unsprung masses, inertia tensors and measured center of gravity.

“Not verified” means this bounded public source audit did not establish the value, not that the information cannot exist elsewhere. The [Ducati manual selector](https://www.ducati.com/ww/en/service-maintenance/owner-manuals) was opened; no exact 2026 PDF was obtained. Direct HTML fetch returned HTTP 403. The [Öhlins document library](https://www.ohlins.com/en-us/document-library) did not expose an applicable OEM setup document through its fetched text.

The [Öhlins NIX30 aftermarket cartridge listing](https://www.ohlins.com/en-us/motorcycle-suspension/supersport/ducati-899-959-panigale-panigale-v4-(showa)?v=ducati-streetfighter-v4-2024) is explicitly a replacement cartridge application associated with earlier motorcycles and Showa forks. Its settings must not become 2026 OEM S constants. No third party advertised spring rate or generic sag target was imported.

## Next implementation boundary

Build contact release, landing and suspension tests with explicitly synthetic parameters first. For calibrated bike behavior, obtain the exact manual and installed component identifiers, measure axle loads and sag under known loading, obtain spring and damper bench curves, and measure linkage geometry. Travel alone cannot determine spring stiffness or damping. A fixed passive damper prototype may be useful, but must remain labeled separately from Smart EC 3.0 until its control behavior is supported by data.
