extends RefCounted
## Preserve rigid rider shadows when a camera excludes the rider's surface layer.
## Shared meshes inherit articulation from their parent, with no duplicate pose state.


static func install(root: Node) -> Array[MeshInstance3D]:
	assert(not root.has_node("RiderShadow"), "Rider shadows already installed")
	var casters: Array[MeshInstance3D] = []
	for child in root.get_children():
		casters.append_array(install(child))
	if root is MeshInstance3D and root.mesh != null:
		if root.cast_shadow == GeometryInstance3D.SHADOW_CASTING_SETTING_OFF:
			return casters
		var shadow := MeshInstance3D.new()
		shadow.name = "RiderShadow"
		shadow.mesh = root.mesh
		shadow.material_override = root.material_override
		for surface in root.mesh.get_surface_count():
			shadow.set_surface_override_material(
				surface, root.get_surface_override_material(surface)
			)
		shadow.layers = 1
		shadow.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_SHADOWS_ONLY
		shadow.gi_mode = GeometryInstance3D.GI_MODE_DISABLED
		root.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		root.add_child(shadow)
		casters.append(shadow)
	return casters
