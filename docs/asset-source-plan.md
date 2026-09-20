# Asset source plan

Verified 19 September 2026. This is a source inventory for the first playable section. Their existence and licenses were checked on their publishers' pages. Asphalt, soil, dry grass, and the sky panorama are now imported and rendered in Godot. Their appearance remains an artistic match rather than a calibrated reconstruction of Thunderhill lighting.

## Material inventory

| Need | Verified reusable source | Implementation decision |
| --- | --- | --- |
| Fresh racing asphalt | [ambientCG Asphalt 010](https://ambientcg.com/view?id=Asphalt010), CC0 | Start here: publisher tags identify clean, dark, flat asphalt. Tune aggregate scale and roughness against the repave reference. Do not infer grip from appearance. |
| Bare earth beside pavement | [Poly Haven Brown Mud Dry](https://polyhaven.com/a/brown_mud_dry), Rob Tuytel, CC0 | Blend into the shoulder using a painted surface mask. Published physical tile width is 1.3 m. Match color and stone density after footage inspection. |
| Dry grass ground | [Poly Haven Withered Grass](https://polyhaven.com/a/withered_grass), Charlotte Baglioni, CC0 | Candidate for summer terrain, mixed with exposed soil to avoid an uninterrupted carpet. Published tile width is 2 m. |
| Gravel | [Poly Haven Gravel Stones](https://polyhaven.com/a/gravel_stones), Amal Kumar, CC0 | Use only where reference confirms gravel. Published tile width is 2 m; the source is coarse grey gravel, not proof of Thunderhill's runoff composition. |
| Nearby grass silhouettes | [Poly Haven Grass Medium 01](https://polyhaven.com/a/grass_medium_01), Rico Cilliers and Rob Tuytel, CC0 | Includes a dry diffuse variant. Bake or reduce to sparse cards and simplified clumps before use. The source page lists 2 million triangles, so importing the full asset as every grass instance is unsuitable. |
| Sky lighting reference | [Poly Haven Kloofendal 48d Partly Cloudy Pure Sky](https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky), Greg Zaal and Jarod Guest, CC0 | Sky only image avoids importing an unrelated landscape skyline. It is a lighting candidate, not a photograph of Thunderhill. Match sun direction and cloud cover to the selected reference session. |

[Poly Haven Asphalt 02](https://polyhaven.com/a/asphalt_02) is also CC0, but its weathering and cracks make it a poor default for newly repaved racing pavement. Keep it out of the initial racing surface.

Both [ambientCG's license](https://docs.ambientcg.com/license/) and [Poly Haven's license](https://polyhaven.com/license) permit modifying and redistributing the asset files under CC0. Keep asset licenses separate from the repository's MIT code license. Poly Haven's website text, previews, and other website content are not automatically covered by its asset license. Use the downloadable asset files for shipped materials.

## Motorcycle mesh search, 20 September 2026

No current generation Streetfighter mesh with verified unrestricted source
redistribution rights was found in this search. This is a search result, not
proof that no such asset exists.

The [OUTPISTON model](https://sketchfab.com/3d-models/2024-ducati-streetfighter-v4-s-c501252f8af64c559bf91dc306c3a550)
is explicitly labeled 2024. Its [publisher API](https://api.sketchfab.com/v3/models/c501252f8af64c559bf91dc306c3a550)
reports 102,000 faces and CC Attribution NonCommercial ShareAlike 4.0.
That license permits redistribution under its restrictions, but does not make
the mesh an unrestricted open asset or MIT licensed content. Its generation
also needs visual verification because the description mixes the 2024 label
with specifications associated with the newer motorcycle. It was not imported.

Continue authoring original geometry against the onboard footage and Ducati
manufacturer references. A marketplace render or availability of a download
does not establish source redistribution permission.

## Original assets to author

Create curb meshes, edge paint, braking boards, simple barriers, and buildings from documented dimensions and observed proportions. Model their visible layout from the reference collection. Author neutral markings until any specific graphics have a verified source. Keep curb collision geometry separate from texture detail; a normal map must not stand in for a curb's physical height.

The user has selected the 2026 Ducati Streetfighter V4 S. The [motorcycle reference](motorcycle-reference.md) records verified current generation manufacturer specifications, regional differences, and the physical parameters still missing. The full metadata of [Ken Moto's reference video](https://www.youtube.com/watch?v=yVjzZqYbKuM) also identifies a 2026 Ducati Streetfighter V4S. That establishes the model shown, but does not establish its tire choice, tuning, gearing, rider mass, or suspension setup.

## How the videos will be used

Use the videos to identify scenery, surface boundaries, colors, landmarks, and camera sightlines. Record a source URL and timestamp for each observation, with a confidence note. A footage based estimate is not a measured dimension.

Extracting frames for publicly distributed textures requires a verified reuse basis for that footage. [YouTube's official license guidance](https://support.google.com/youtube/answer/2797468?hl=en) distinguishes the default standard license from Creative Commons Attribution. No reusable license has been verified for the user video, and it should not be bundled into the repository. Viewing reference footage does not establish redistribution rights. CC0 materials and original geometry allow work to continue without relying on those extracted pixels.

Even with reusable footage, perspective, lens distortion, baked lighting, compression, and motion blur can make road frames unsuitable as texture maps. Prefer the listed PBR sources for material detail. Use footage for matching composition and appearance rather than promising automatic asset reconstruction.

## Mac implementation choices

Begin material imports at 2K and inspect at actual riding camera distance. Use normal and roughness maps for fine detail, with geometry reserved for silhouettes and physically meaningful surface changes. Mipmaps, bounded vegetation draw distance, and simplified distant geometry are part of the initial implementation. These are starting choices, not measured performance claims.

Store surface identity independently of material color so asphalt, curb, dirt, and gravel retain explicit physics semantics when art changes. Keep the source asset, conversion settings, resulting checksum, author, and license in an asset manifest when files are imported. Compare the same reference camera view before and after material changes, and profile on the target Mac before expanding the full circuit.

## Acquired samples and visual inspection

Run `uv run tools/fetch_materials.py` to acquire the pinned 1K samples from [the candidate manifest](../data/reference/material-candidates.json). The script verifies SHA256 for every download and publisher MD5 where supplied, extracts only color, OpenGL normal, and roughness maps from the asphalt archive, and creates `artifacts/reference/materials/candidate-sheet.jpg`. The raw archive and six individual downloads total 14,695,643 bytes. Raw assets and the sheet stay in ignored artifacts; the source URLs, authors, licenses, hashes, and acquisition script are tracked. Explicit `--refresh-manifest` resolves current publisher metadata and creates new pins for review.

The acquired color sheet was visually inspected. Asphalt010 has a fine dark grey aggregate with no dominant cracks, making it a plausible repave starting point. Brown Mud Dry contains pronounced coarse stones and tracks; it should not cover every shoulder uniformly. Withered Grass is pale beige and dense, so blending exposed ground and adding sparse silhouettes will be necessary. These observations concern the samples only. They do not establish a calibrated color match, actual aggregate size at Thunderhill, or correct lighting response in Godot. A sheet of color maps also cannot validate the normal and roughness response at a low riding camera angle.

The pinned acquisition was run a second time successfully, verifying all downloaded hashes and regenerating the sheet from the cached files.

## Near vegetation implementation

`tools/fetch_grass.py` acquires pinned CC0 Grass Medium 01 source files and extracts three complete author tufts, preserving mesh shape and atlas coordinates. The source dry diffuse RGB and alpha mask form one 1K RGBA atlas. Mesh triangles, conversion details, authors, licenses and source hashes are stored in `godot/assets/grass/grass.json`. This JSON is explicitly included in native export presets.

Godot instantiates these tufts in spatial patches with bounded visibility. Placement and scale remain artistic estimates. Broad terrain color variation uses several spatial scales instead of a repeating sine pattern; this changes appearance only, not surface identity or tire grip.
