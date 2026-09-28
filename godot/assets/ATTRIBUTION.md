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


## Canopy study diffuse lighting

The Burley diffuse implementation in `shaders/canopy_light_study.gdshaderinc`
is adapted from Godot Engine at revision `ed1daf0bf`:
https://github.com/godotengine/godot/blob/ed1daf0bf/servers/rendering/renderer_rd/shaders/scene_forward_lights_inc.glsl
The experimental backscatter lobe is original project code.

Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


`materials/road_scuff_v1.png` is an original generated grayscale deposit mask,
used under the project MIT license. The built in image generation tool received
text only. Exact prompt and source hash are in `materials/road_scuff_v1.json`.
It is a numeric coverage texture with ordinary numeric mipmaps, not an sRGB
albedo or a measured scan. Placement and dimensions are artistic estimates.

<!-- motogp-circuits:begin (tools/build_circuit.py docs) -->

## MotoGP circuits

`godot/tracks/<id>/track.json` and the generated meshes under `godot/tracks/<id>/generated/` are
derived geographic databases under ODbL 1.0 (OpenStreetMap contributors,
https://www.openstreetmap.org/copyright), separate from the MIT game code. The orthoimagery
tiles and elevation in `generated/` are not committed; `tools/build_circuit.py` fetches them
from the sources below, whose licences and attributions apply. Full provenance, pinned URLs
and response hashes: docs/motogp-circuits.md. The project is not affiliated with MotoGP,
Dorna or any circuit.

* `buriram` (Chang International Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `goiania` (Autódromo Internacional de Goiânia – Ayrton Senna): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `cota` (Circuit of the Americas): OpenStreetMap contributors (ODbL 1.0); USGS, USDA, The National Map: Orthoimagery (NAIP) (Public domain (U.S. Government work), https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits); U.S. Geological Survey, 3D Elevation Program (Public domain (U.S. Government work), https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits).
* `jerez` (Circuito de Jerez - Angel Nieto): OpenStreetMap contributors (ODbL 1.0); PNOA cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT05 cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT25 cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
* `le-mans` (Le Mans Bugatti Circuit): OpenStreetMap contributors (ODbL 1.0); IGN - BD ORTHO (Licence Ouverte / Open Licence Etalab 2.0, https://www.data.gouv.fr/pages/legal/licences/etalab-2.0); IGN - MNT LiDAR HD (Licence Ouverte / Open Licence Etalab 2.0, https://www.data.gouv.fr/pages/legal/licences/etalab-2.0); IGN - RGE ALTI (Licence Ouverte / Open Licence Etalab 2.0, https://www.data.gouv.fr/pages/legal/licences/etalab-2.0).
* `catalunya` (Circuit de Barcelona-Catalunya): OpenStreetMap contributors (ODbL 1.0); PNOA cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT05 cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT25 cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
* `mugello` (Autodromo Internazionale del Mugello): OpenStreetMap contributors (ODbL 1.0); Ortofoto 2023 - Fonte dei dati: Regione Toscana - Base Informativa Territoriale regionale, art. 55 della L.R. 65/2014 (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); Altimetria 10 metri, DTM orografico - fonte Regione Toscana - SIPT (da CTR 1:10.000) (CC BY (Regione Toscana open data, dataset dem10mt), https://dati.toscana.it/dataset/dem10mt).
* `balaton-park` (Balaton Park Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `brno` (Automotodrom Brno): OpenStreetMap contributors (ODbL 1.0); © ČÚZK (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
* `assen` (TT Circuit Assen): OpenStreetMap contributors (ODbL 1.0); Beeldmateriaal Nederland / PDOK (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); AHN / PDOK (CC0 1.0, http://creativecommons.org/publicdomain/zero/1.0/deed.nl).
* `sachsenring` (Sachsenring): OpenStreetMap contributors (ODbL 1.0); Quelle: GeoSN, dl-de/by-2-0 (Datenlizenz Deutschland - Namensnennung - Version 2.0 (dl-de/by-2-0), https://www.govdata.de/dl-de/by-2-0); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `silverstone` (Silverstone Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM); © Environment Agency copyright and/or database right 2022. All rights reserved. (Open Government Licence v3.0, https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
* `aragon` (MotorLand Aragón): OpenStreetMap contributors (ODbL 1.0); PNOA cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT05 cedido por (c) Instituto Geografico Nacional de Espana (CC BY 4.0); MDT25 cedido por (c) Instituto Geografico Nacional de Espana (CC BY 4.0).
* `misano` (Misano World Circuit Marco Simoncelli): OpenStreetMap contributors (ODbL 1.0); Ortofoto RER 2023-24 RGB - Regione Emilia-Romagna, Archivio Cartografico (CC BY 4.0) (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); DTM 0,5x0,5m RER 2023-24 - Regione Emilia-Romagna, Archivio Cartografico (CC BY 4.0) (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `red-bull-ring` (Red Bull Ring): OpenStreetMap contributors (ODbL 1.0); Datenquelle: basemap.at (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); © BEV (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `motegi` (Mobility Resort Motegi (road course)): OpenStreetMap contributors (ODbL 1.0); 出典：国土地理院 地理院タイル（全国最新写真（シームレス））を加工して作成 https://maps.gsi.go.jp/development/ichiran.html (Public Data License 1.0 (PDL1.0, 公共データ利用規約 第1.0版; compatible with CC BY 4.0), https://www.digital.go.jp/resources/open_data/public_data_license_v1.0); 出典：国土地理院 地理院タイル（標高タイル（基盤地図情報数値標高モデル））を加工して作成 https://maps.gsi.go.jp/development/ichiran.html (Public Data License 1.0 (PDL1.0, 公共データ利用規約 第1.0版; compatible with CC BY 4.0), https://www.digital.go.jp/resources/open_data/public_data_license_v1.0).
* `mandalika` (Pertamina Mandalika International Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `phillip-island` (Phillip Island Grand Prix Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); © Commonwealth of Australia (Geoscience Australia) (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `sepang` (Petronas Sepang International Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `lusail` (Lusail International Circuit): OpenStreetMap contributors (ODbL 1.0); Contains modified Copernicus Sentinel data 2026 (Copernicus Sentinel data terms (free, full and open), https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `portimao` (Autódromo Internacional do Algarve): OpenStreetMap contributors (ODbL 1.0); Ortofotos 2025: informação geográfica propriedade da Direção-Geral do Território (DGT), CC BY 4.0 (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); Copernicus DEM GLO-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA (Copernicus DEM licence (free, including commercial use), https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).
* `valencia` (Circuit Ricardo Tormo): OpenStreetMap contributors (ODbL 1.0); PNOA cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT05 cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/); MDT25 cedido por © Instituto Geográfico Nacional – CC BY 4.0 scne.es (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).
* `laguna-seca` (WeatherTech Raceway Laguna Seca): OpenStreetMap contributors (ODbL 1.0); USGS, USDA, The National Map: Orthoimagery (NAIP) (Public domain (U.S. Government work), https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits); U.S. Geological Survey, 3D Elevation Program (Public domain (U.S. Government work), https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits).

<!-- motogp-circuits:end -->
