# Asset attribution

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
