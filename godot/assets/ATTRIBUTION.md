# Asset attribution

The material JPG files in `materials/` are CC0 assets, separate from the MIT game code. Their pinned source hashes are in `data/reference/material-candidates.json` at the repository root.

* Asphalt010, Lennart Demes / ambientCG: https://ambientcg.com/a/Asphalt010 . CC0: https://docs.ambientcg.com/license/
* Withered Grass, Charlotte Baglioni / Poly Haven: https://polyhaven.com/a/withered_grass . CC0: https://polyhaven.com/license
* Brown Mud Dry, Rob Tuytel / Poly Haven: https://polyhaven.com/a/brown_mud_dry . CC0: https://polyhaven.com/license

The geographic files in `godot/data/` are derived geographic databases under ODbL 1.0, with attribution to OpenStreetMap contributors, USGS 3DEP, and USDA NAIP. Their manifests carry provenance and limitations. https://www.openstreetmap.org/copyright

Motorcycle and scenery geometry are original procedural visual approximations. No Ducati factory CAD, commercial simulator meshes, or video pixels are included. The project is not affiliated with Ducati or Thunderhill Raceway.

The HDR panorama in `sky/kloofendal_2k.hdr` is Kloofendal 48d Partly Cloudy Pure Sky, Greg Zaal and Jarod Guest / Poly Haven, CC0: https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky . License: https://polyhaven.com/license . Download hash and measured brightest pixel direction are in `godot/data/sky.json`. This is a generic sky, not a reconstruction of Thunderhill weather.

Grass tuft meshes and `grass/dry_grass_rgba.png` derive from Grass Medium 01, modeled by Rico Cilliers with photography by Rob Tuytel, Poly Haven: https://polyhaven.com/a/grass_medium_01 . CC0: https://polyhaven.com/license . `tools/fetch_grass.py` extracts three complete source tufts with their original UVs and combines the dry diffuse and alpha textures. Source URLs, publisher MD5s and downloaded SHA256 hashes are pinned in `grass/grass.json`. Runtime scaling and decorative placement are artistic approximations, not surveyed plant locations. This paragraph qualifies the earlier statement about original procedural scenery.
