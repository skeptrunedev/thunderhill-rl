# Ground treatment experiments

Three independent treatments were rendered at the same Turn 2 apex pose and
lighting. Their original standalone variants remain developer experiments.
The regional generated variant described below is now the default game treatment.
The motorcycle asset work is deferred while the owner obtains its download.

| Treatment | Implementation | Visible result |
| --- | --- | --- |
| Generated dry field | Original larger straw and soil texture, standalone material without production aerial or procedural bands | Strongest directional improvement. Continuous dry straw coverage removes the smooth brown apron. Too uniformly dense, with insufficient larger disturbed patterns. |
| Scanned turf | Poly Haven Grass Ground color, OpenGL normal and roughness maps | Natural small scale variation but wrong green vegetation state. Reject this particular source as a Thunderhill match. |
| Physical fragments | 9000 deterministic clumps, actual terrain placement, fine curved strips and sparse stems with shadows | Adds local depth but does not resolve the dominant underlying material composition. First bright oversized version was visibly poor; refined fragments are less conspicuous. |

Root and independent review agree on this ranking. Reference camera, exposure
and motion blur differ; no exact material or straw scale match is established.
The generated treatment also replaces distant ground and access areas, so it
needs regional composition before becoming a production material.

## Evidence

* `artifacts/radical-ground-comparison/00.png`: production versus generated.
* `artifacts/radical-ground-comparison/01.png`: production versus scan.
* `artifacts/radical-ground-comparison/02.png`: production versus refined geometry.
* `artifacts/radical-ground-generated.mp4`: generated treatment on the same
  compatible recorded Turn 2 trajectory, 2699 transitions completed.
* `artifacts/radical-ground-generated-review`: twelve sampled video frames.

The 700 frame video is 1280 by 800 at 30 FPS, including a short stationary tail.
Fixed rate movie capture is not a live performance benchmark. Sampled frames
show continuous ground coverage through the bend, but do not establish fine
motion stability. Native Mac performance of these experiments is unverified.

## Reproduction

Use a real renderer and the pinned Godot executable from the project setup.
From the repository root, replace `godot` below with that executable:

```sh
godot --path godot --script res://tools/preview_lighting.gd -- \
  --production-only --camera=2 --station-m=1150 --lateral-m=4 \
  --lean-deg=-25 --view-yaw-deg=8 --ground-study=generated \
  --output-dir=/absolute/new/output
```

Choose `generated`, `scan`, `geometry`, or `production`. Metadata records the
chosen treatment and source parameters. Production terrain metadata is retained
separately for comparison rather than incorrectly reading it from the replacement.
The scan helper loads pinned source maps in the developer workspace; its loose
image loading has not been adapted or verified for an exported game.

```sh
godot --path godot --script res://tools/preview_ground_replay.gd \
  --write-movie /absolute/new/movie.avi --fixed-fps 30 --quit-after 700 -- \
  --ground-study=generated --replay=/absolute/compatible/replay.jsonl \
  --preview-camera=2
```

The replay study rejects agent and control test sessions. It changes appearance
only and does not produce a learned policy or establish physical accuracy.

## Sources and validation

The generated image, exact prompt and limitations are in
`assets/source/ground-studies`. Built in image generation produced 1254 square
pixels; color mipmaps use linear light filtering. The source is original artwork,
not extracted footage or a measured scan.

The scanned material is [Grass Ground](https://polyhaven.com/a/grass_ground)
by Charlotte Baglioni, [CC0](https://polyhaven.com/license). Exact download URLs,
publisher MD5 and acquired SHA256 hashes are in
`godot/assets/materials/scan_study/source.json`.

All three treatments compiled and rendered with Vulkan Mobile. Full project
editor import completed without script errors. The geometry helper was checked
for finite upward normals, terrain placement and road clearance. It contains
1,008,000 total triangles, culled in spatial patches beyond 65 metres; that count
is not a performance guarantee. The original comparisons predate regional integration below.

## Regional integration

The generated material now replaces only the pale Turn 2 field annotation.
The darker western annotation, distant ground and explicit bare access corridors
keep their existing materials. A bounded five sample maximum extends the pale
mask toward the shoulder, using offsets of three metres in the two map axes.
This is a directional artistic expansion, not a uniform measured dilation.
Region strength and texture bounds are respected.

Albedo, roughness and independently differentiated relief gradients blend under
the same mask. Differentiating heights before blending avoids adding artificial
ridges at region boundaries. The extra source is loaded as a baked texture
resource, so the production path does not depend on loose developer image files.

Root and independent review prefer the regional apex over blanket replacement:
it retains the new foreground while preserving outer terrain differences. The
source still lacks some larger disturbed straw patterns from the footage.

`--ground-study=regional` explicitly selects this path; `production` uses the
current default and `legacy` disables it for static comparisons. The geometry
study explicitly uses the earlier underlying material to preserve its original
isolation. Both existing scenery resources were rebaked after the track setup
changed. Track, terrain, collision and friction data are unchanged.

The rendered human controls check passed with zero failures. Linux frame
intervals were 17.36 ms median and 17.92 ms p95 across 265 samples. These are
local short test results, not native Mac performance measurements.

Five real agent camera captures at 640 by 360 passed PNG and hash validation,
immutable artifacts, frozen tick and deterministic idle pose, changed pixels
after advance, queued advance/reset serialization, no privileged telemetry,
and headless rejection. Evidence: `artifacts/regional-ground-camera-qa`.

The actual default material is captured in
`artifacts/regional-ground-production-apex/production.png` and compared with
the previous material in `artifacts/regional-ground-comparison/00.png`.
`artifacts/regional-ground-production.mp4` records the final default over the
same 2699 replay transitions, replacing the earlier blanket material video as
the current game review clip. Mac export `aab7e3af3554-46126a98593b` succeeded;
the current laptop was offline at verification, so native review is pending.
