class_name CircuitImagery
extends RefCounted
## Orthoimagery materials for MotoGP circuits built by tools/build_circuit.py.
## The images live in the ignored generated/ folder (with .gdignore), so they are
## decoded at runtime instead of imported. Headless sessions draw nothing and skip
## the decode entirely; the materials still exist so scene construction is shared.

const SHADER := preload("res://shaders/circuit_ground.gdshader")
## Imagery pixels coarser than this cannot resolve the road; pavement then uses the
## generic asphalt material instead of photographing itself from ten metres up.
const IMAGED_PAVEMENT_MAX_PIXEL_M := 1.0

var error := ""
var ground: ShaderMaterial
var pavement: ShaderMaterial


func configure(generated_dir: String, track_sha256: String) -> String:
	var path := generated_dir + "imagery.json"
	if not FileAccess.file_exists(path):
		return "Missing " + path
	var manifest: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not manifest is Dictionary or manifest.get("schema_version") != 1:
		return "Invalid circuit imagery manifest"
	if manifest.get("track_sha256") != track_sha256:
		return "Circuit imagery was built for a different track.json"
	ground = _material(false)
	pavement = _material(true)
	var detail: Dictionary = manifest.detail
	var imaged := float(detail.resolution_m) <= IMAGED_PAVEMENT_MAX_PIXEL_M
	pavement.set_shader_parameter("imaged_pavement", imaged)
	if DisplayServer.get_name() == "headless":
		return ""
	var images: Array[Image] = []
	var grid: Dictionary = detail.grid
	var index := Image.create_empty(int(grid.ncols), int(grid.nrows), false, Image.FORMAT_R8)
	for layer: Dictionary in detail.layers:
		var image := _load(generated_dir + "imagery/" + str(layer.file))
		if image == null:
			return "Cannot read circuit imagery " + str(layer.file)
		images.append(image)
		var key: Array = layer.key
		var col := int(key[0]) - int(grid.col0)
		var row := int(grid.row_top) - int(key[1])
		index.set_pixel(col, row, Color(float(images.size()) / 255.0, 0, 0))
	if images.size() > 254:
		return "Too many circuit detail tiles for the eight bit index"
	var overview := _load(generated_dir + "imagery/" + str(manifest.overview.file))
	var horizon := _load(generated_dir + "imagery/" + str(manifest.horizon.file))
	if overview == null or horizon == null:
		return "Cannot read circuit overview or horizon imagery"
	for material: ShaderMaterial in [ground, pavement]:
		if not images.is_empty():
			var array := Texture2DArray.new()
			var created := array.create_from_images(images)
			if created != OK:
				return "Cannot create circuit detail texture array: %s" % created
			material.set_shader_parameter("detail_tiles", array)
			material.set_shader_parameter("tile_index", ImageTexture.create_from_image(index))
			material.set_shader_parameter("has_detail", true)
		material.set_shader_parameter("grid_origin", Vector2(grid.x0, grid.z0))
		material.set_shader_parameter("grid_cells", Vector2(grid.ncols, grid.nrows))
		material.set_shader_parameter("tile_m", float(grid.tile_m))
		material.set_shader_parameter("overview", ImageTexture.create_from_image(overview))
		material.set_shader_parameter("overview_bounds", _bounds(manifest.overview.bounds))
		material.set_shader_parameter("horizon_image", ImageTexture.create_from_image(horizon))
		material.set_shader_parameter("horizon_bounds", _bounds(manifest.horizon.bounds))
	return ""


func _material(is_pavement: bool) -> ShaderMaterial:
	var material := ShaderMaterial.new()
	material.shader = SHADER
	material.set_shader_parameter("pavement", is_pavement)
	material.set_shader_parameter(
		"asphalt_color", load("res://assets/materials/Asphalt010_1K-JPG_Color.jpg")
	)
	material.set_shader_parameter(
		"asphalt_normal", load("res://assets/materials/Asphalt010_1K-JPG_NormalGL.jpg")
	)
	material.set_shader_parameter(
		"ground_detail", load("res://assets/materials/brown_mud_dry_diff_1k.jpg")
	)
	return material


static func _bounds(b: Dictionary) -> Vector4:
	return Vector4(float(b.x0), float(b.z0), float(b.x1), float(b.z1))


static func _load(path: String) -> Image:
	var image := Image.load_from_file(path)
	if image == null or image.is_empty():
		return null
	image.convert(Image.FORMAT_RGB8)
	image.generate_mipmaps()
	return image
