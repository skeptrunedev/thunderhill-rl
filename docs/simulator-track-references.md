# Existing simulator track audit

Checked 19 September 2026. Existing Thunderhill implementations do exist, but this audit found no verified accurate East mesh with permission to redistribute in our open source game, and no verified version representing the 2026 repave. The two strongest leads solve different parts of the problem.

## Best fidelity lead: Nukedrop for Assetto Corsa

[Thunderhill v13](https://www.patreon.com/nukedrop/posts/thunderhill-v13-81182397), released 7 April 2023, is the strongest established game implementation found. The author's [February 2021 explanation](https://www.patreon.com/nukedrop/posts/february-2021-47685128) describes correcting the terrain using publicly available US LiDAR and acknowledges that the earlier modeled hill was too shallow. Later [LiDAR updates](https://www.patreon.com/nukedrop/posts/thunderhill-part-49386980) include West. V13 addresses seams and overly severe curbs.

This supports a LiDAR informed implementation, not a verified millimeter accurate survey. The source survey, numerical error, and exact v13 course inventory were not available in the inspected material. The release post is locked, and no license allowing a Godot port or public mesh redistribution was found. Nothing was purchased or downloaded. Internal file formats remain uninspected; do not claim that an editable Blender or FBX source is available.

The practical next step is an author request for the East source mesh, survey provenance, and explicit permission to redistribute a Godot adaptation. Buying access to an Assetto Corsa mod would not by itself establish that permission. No message has been sent.

## Best structured geometry lead: CARLA OpenDRIVE

[Wolverine CARLA Thunderhill](https://github.com/szu-zy/Wolverine_CARLA_Thunderhill) actually publishes [Thunderhill_origin.xodr](https://github.com/szu-zy/Wolverine_CARLA_Thunderhill/blob/main/Thunderhill_origin.xodr), another XODR version, and waypoint data. The README describes OpenStreetMap and MathWorks RoadRunner construction, followed by driving in CARLA 0.9.10. It does not claim a laser survey and notes that road apexes need improvement.

The XML was inspected directly: OpenDRIVE 1.4, MathWorks vendor, 11 May 2021 timestamp, 24 road elements, 75 elevation records, and 24 superelevation records. Road element lengths sum to approximately 4,585 m. That sum is not automatically a lap length, and records do not prove measured elevation or camber. The README points to the East area, but the exact bypass configuration needs geometric comparison.

This is the most convenient existing road representation to evaluate for conversion. It is not presently cleared for inclusion: GitHub reports no license, the README links an unavailable original repository, and the OSM derivation and any RoadRunner asset provenance need resolution. Public source visibility is not an open source grant. The XML was parsed in memory for inspection and was not copied into our tracked assets.

Further source inspection rules this map out as a physical geometry authority. In both published XODR files, every elevation polynomial and every superelevation polynomial has all coefficients equal to zero. The actual elevation range is therefore 0 to 0 m and bank angle is 0 throughout. Each of the 24 roads has one driving lane of constant 10 m width, flanked by 0.635 m shoulders and 2 m sidewalks. Tiny polynomial coefficients on a few widths are floating point noise, not meaningful measured width variation. These are uniform authored cross sections, not evidence of surveyed pavement edges.

The origin file declares a transverse Mercator georeference with `lat_0=0.00929897847085288` and `lon_0=0.01164650401068383`; the second file uses similarly near zero coordinates. Neither is a Thunderhill geographic origin. Geographic alignment needs a separate transform and cannot be accepted from this header.

A local comparison plot sampled the actual line, arc, parametric cubic, and spiral road geometry and compared it visually with our East OSM outline. The shapes clearly correspond to East, with broadly similar proportions. The sampled XODR extent is about 843 by 1,028 local units, compared with about 840 by 1,022 m for the acquired OSM route projected into EPSG:26910. This establishes approximate plan correspondence, not surveyed alignment: each panel is independently fitted and there is no calibrated residual measurement. The sharper inward kink visible on the OSM outline is smoothed in the CARLA geometry. No alternative Cyclone/bypass branch is represented, and the flat elevations cannot represent the Cyclone's vertical profile at all. Exact turn 5 route identity remains unverified; do not label this a validated Cyclone reconstruction.

The inspection plot is local only at `artifacts/reference/sim-track/wolverine-outline-comparison.png`. It was visually inspected. This result downgrades the candidate to an example of flat East topology and road generation, not a shortcut to realistic width, elevation, or camber.

## Open source prototype lead: SDSandbox

[SDSandbox](https://github.com/tawnkramer/sdsandbox) has a permissive BSD style three clause [repository license](https://github.com/tawnkramer/sdsandbox/blob/ec04c400f045222375ec82e649bb051fb17aad41/LICENSE), an actual Thunderhill Unity scene, terrain, and XYZ path files. Small relevant files were acquired at commit `ec04c400f045222375ec82e649bb051fb17aad41` into ignored `artifacts/reference/sim-track`; the manifest records their hashes. No complete game or large terrain image was downloaded.

The scene explicitly loads `thunderhill2`. That file contains 1,224 XYZ samples; consecutive point distances sum to about 2,825 Unity coordinate units. The preview was visually inspected and appears to show West, with artificial continuous barriers around the circuit. This is a visual inference, not confirmation from its author. Neither scale calibration nor survey provenance was found. Scene changes inspected date to 2021. It is not suitable as our East geometry authority. The code and path representation can inform prototyping; aerial imagery and bundled third party assets require their own provenance review before reuse.

## Other candidates

| Candidate | Concrete finding | Decision |
| --- | --- | --- |
| [Good Luck League Thunderhill Skidpad](https://github.com/MeAndMyPenguin/GLL_Thunderhill_Skidpad/releases) | Author linked GitHub release 1.0.1, 28 June 2026; 217,704,959 byte 7z archive; no repository license | Skidpad only, not East. No download. |
| [GTR2 Thunderhill long 0.9](https://esport-racing.de/thunderhill-raceway-park/) | Distributed author notes from November 2008 describe fictional scenery, modeled hills, and positioning defects; credits trace through GPL and rFactor conversions | Reject as an accuracy source. Specific past conversion permission is not a general reuse license. |
| LGSVL student reconstruction | [Project repository](https://github.com/jeniwang/DSC-180B-Team6) documents a Unity track, but inspected tree contains analysis, logs, and report rather than the Unity map; no license | No usable track source located. |

Searches also covered rFactor 2, BeamNG, iRacing, and motorcycle simulator references. No stronger primary source with reusable East geometry was verified. This is a search result, not a claim that no private or obscure implementation exists.

Keep the existing USGS and OSM reconstruction route active while evaluating author supplied mesh access. A licensed historical mesh could accelerate scenery and topology work, but the current curb profiles, pavement limits, and surface appearance would still need the repave reference pass.

Machine readable evidence and acquisition hashes: [simulator track sources](../data/reference/simulator-track-sources.json).
