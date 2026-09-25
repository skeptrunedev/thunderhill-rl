extends "res://tests/test_wall_contact.gd"
const BASELINE := preload("res://tests/baseline_bike_sweep.gd")
const CANDIDATE := preload("res://tests/candidate_bike_sweep.gd")
func check_historical_wall(game: Node3D) -> void:
	var envelope: RefCounted = game.collision_sweep.envelope
	for round_index in 3:
		var order: Array = [["baseline", BASELINE], ["candidate", CANDIDATE]]
		if round_index % 2 == 1:
			order.reverse()
		for entry in order:
			var solver: RefCounted = entry[1].new()
			check(solver.build(envelope).is_empty(), "A/B solver build failed")
			game.collision_sweep = solver
			print("WALL_AB_BEGIN round=", round_index, " solver=", entry[0])
			await super.check_historical_wall(game)
			print("WALL_AB_END round=", round_index, " solver=", entry[0])
