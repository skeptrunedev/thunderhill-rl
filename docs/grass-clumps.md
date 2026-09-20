# Dry grass clumps

The source Poly Haven assets contain individual stems and tiny tufts. Their unscaled heights are approximately 0.323, 0.112 and 0.049 metres. Treating each as a whole roadside clump left the visible shoulders almost bare.

The scene now assembles three reusable clump variants. Each contains one tall, two small A and three small E source meshes, with independent rotations, offsets and modest scaling. Each clump has 634 triangles. Original source UVs, normals, dry color and alpha are retained. Authorship, CC0 licensing and source hashes remain in `godot/assets/grass/grass.json`.

Placement targets 96,000 accepted clumps with a bounded attempt count, concentrated near the circuit. This is decorative density, not a vegetation survey. Roots follow the sampled visible terrain normal and height. A clump spanning multiple terrain triangles remains a local tangent plane approximation. Mapped paving and buildings exclude clumps with a 0.4 metre margin; the main road retains its existing clearance.

Spatial MultiMesh batches retain the 80 metre visibility limit and disabled shadows. The higher density materially increases the potentially visible triangle count. A rendered control diagnostic at 1920 by 1200 on the local RTX 2080 Ti passed with zero failures, with 267 frame intervals reporting median 17.241 ms and 95th percentile 17.850 ms. This brief diagnostic is not whole circuit or native Mac performance acceptance.

The straight preview confirms more visible grouped vegetation. The broad terrain color, scenery, motorcycle dynamics and overall photographic realism still require further work. No training has started.
