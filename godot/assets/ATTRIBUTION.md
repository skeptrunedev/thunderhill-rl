# Asset attribution

`materials/racing_asphalt_v1.png` is an original generated asphalt albedo.
Its exact prompt, source hash and artistic shader estimates are recorded in
`assets/source/material-studies/README.md`. It contains no extracted video
pixels and is not a measured scan. It replaces the generic Asphalt010 material
in normal gameplay; that material remains available for comparison studies.

The material JPG files in `materials/` are CC0 assets, separate from the MIT game code. Their pinned source hashes are in `data/reference/material-candidates.json` at the repository root.

* Asphalt010, Lennart Demes / ambientCG: https://ambientcg.com/a/Asphalt010 . CC0: https://docs.ambientcg.com/license/
* Withered Grass, Charlotte Baglioni / Poly Haven: https://polyhaven.com/a/withered_grass . CC0: https://polyhaven.com/license
* Brown Mud Dry, Rob Tuytel / Poly Haven: https://polyhaven.com/a/brown_mud_dry . CC0: https://polyhaven.com/license

The geographic files in `godot/data/` are derived geographic databases under ODbL 1.0, with attribution to OpenStreetMap contributors, USGS 3DEP, and USDA NAIP. Their manifests carry provenance and limitations. https://www.openstreetmap.org/copyright

Motorcycle and scenery geometry are original procedural visual approximations. No Ducati factory CAD, commercial simulator meshes, or video pixels are included. The project is not affiliated with Ducati or Thunderhill Raceway.

The HDR panorama in `sky/kloofendal_2k.hdr` is Kloofendal 48d Partly Cloudy Pure Sky, Greg Zaal and Jarod Guest / Poly Haven, CC0: https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky . License: https://polyhaven.com/license . Download hash and measured brightest pixel direction are in `godot/data/sky.json`. This is a generic sky, not a reconstruction of Thunderhill weather.

Grass tuft meshes and `grass/dry_grass_rgba.png` derive from Grass Medium 01, modeled by Rico Cilliers with photography by Rob Tuytel, Poly Haven: https://polyhaven.com/a/grass_medium_01 . CC0: https://polyhaven.com/license . `tools/fetch_grass.py` extracts three complete source tufts with their original UVs and combines the dry diffuse and alpha textures. Source URLs, publisher MD5s and downloaded SHA256 hashes are pinned in `grass/grass.json`. Runtime scaling and decorative placement are artistic approximations, not surveyed plant locations. This paragraph qualifies the earlier statement about original procedural scenery.

`materials/terrain_macro.png` is a processed broad color variation map from USDA NAIP imagery distributed through USGS The National Map. Source tile `m_3912230_sw_10_060_20220715`, acquired July 15, 2022. The [USGS service](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer?f=pjson) identifies NAIP imagery as public domain. `materials/terrain_macro.json` records the source hash, extent, processing and output hash. `tools/build_terrain_color.py` selects dry terrain colors, excludes rejected pixels from smoothing, and encodes restrained relative color gains. The map is photographic appearance under historical illumination, not measured albedo or a surveyed surface classification. No video frames are included.

The rough concrete diffuse, OpenGL normal and roughness JPGs are Rough Concrete by Dimitrios Savva / Poly Haven, CC0: https://polyhaven.com/a/rough_concrete . License: https://polyhaven.com/license . The published tile width is 1.2 metres. Publisher MD5 and acquired SHA256 hashes are pinned in `data/reference/material-candidates.json`. The shader applies paint tint, reduces color contrast and normal intensity, and adds restrained wear. This is a generic material approximation for the pit divider and curbs, not a sampled Thunderhill surface.

The leather normal and roughness JPGs are Leather Red 02 by Rob Tuytel / Poly Haven, CC0: https://polyhaven.com/a/leather_red_02 . License: https://polyhaven.com/license . The published tile width is 0.6 metres. Publisher MD5 and acquired SHA256 hashes are pinned in `data/reference/material-candidates.json`. The original black rider colors are retained, with object space triplanar grain, reduced normal intensity and adjusted roughness. This is a generic leather reference, not a scan of the rider's equipment.

`materials/dry_cut_grass_v1.png` and `materials/dry_cut_grass_v2.png` are original generated grass material studies,
used with custom estimated relief and roughness in `terrain.gdshader`. The exact
prompt, reference use and limitations are recorded in
`assets/source/material-studies/README.md` at the repository root. No source
video frame is redistributed in this texture. It is not a scanned PBR material.

The short bent ribbon blades added by `scenery.gd` are original procedural
geometry, using the generated cut grass albedo. Each clump contains 132 original bent ribbons. Blade dimensions and placement are visual
estimates, not surveyed vegetation.

`materials/fine_shoulder_v1.png` is an original generated fine soil and straw
material guided by the onboard reference. The source, exact prompt and estimated
shader response are documented in `assets/source/material-studies/README.md`.
It is not a measured scan, and no video frame is redistributed in the asset.

The current roadside clump uses 132 original short bent ribbons and the original
generated grass albedo. The earlier two Poly Haven atlas stems are no longer
instantiated. Their source assets and provenance remain archived in the repo.
The terrain shader's world space variation within the grass and soil blend is
original procedural appearance detail, not surveyed vegetation coverage.

`sky/cirrus_patch_v1.png` is an original generated rectilinear cirrus detail view,
used under the project's MIT license. The built in imagegen tool used the supplied
video frame only as a cloud morphology reference. The source image, exact prompt,
hash and authoring limitations are in `assets/source/sky-studies/README.md` at the
repository root. Its `.res` stores color mipmaps filtered in linear light.

`materials/dense_straw_v7.png` is the current original generated straw albedo,
used under the project's MIT license. The image generator received text only.
Source dimensions, SHA256, exact prompt and estimated shader parameters are
recorded in `assets/source/material-studies/README.md` at the repository root.
The sibling `.res` contains linear light filtered color mipmaps. This is an
artist material, not a measured scan or extracted video frame.

Ground experiments only: `materials/scan_study` contains Grass Ground by
Charlotte Baglioni / Poly Haven, CC0. Source: https://polyhaven.com/a/grass_ground
License: https://polyhaven.com/license . Exact publisher URLs, MD5 and SHA256
values are pinned in `materials/scan_study/source.json`. This generic turf is
not a Thunderhill scan.

`materials/ground_study/flattened-field-v1.png` is original generated artwork
under the project MIT license. Source, exact prompt and limitations are in
`assets/source/ground-studies/` at the repository root.

`materials/ground_study/sparse-field-v1.png` is original generated artwork
under the project MIT license. The generator received text only. Exact prompt,
source hash and limitations are in `assets/source/ground-studies/`.
