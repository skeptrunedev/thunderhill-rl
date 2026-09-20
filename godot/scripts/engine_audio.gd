class_name MotorcycleAudio
extends AudioStreamPlayer
## Original synthesized engine and wind preview, not a Ducati recording.
var sim: RefCounted
var muted := true
var phase := 0.0
var wind := 0.0
var random := RandomNumberGenerator.new()
var playback: AudioStreamGeneratorPlayback
const SAMPLE_RATE: float = 22050.0


func _ready() -> void:
	random.seed = 9183
	var generator := AudioStreamGenerator.new()
	generator.mix_rate = SAMPLE_RATE
	generator.buffer_length = 0.08
	stream = generator
	volume_db = -17.0
	play()
	playback = get_stream_playback()


func _process(_dt: float) -> void:
	if playback == null or sim == null:
		return
	var frequency: float = sim.rpm / 60.0 * 2.0
	var level: float = 0.0 if muted else 0.12 + sim.throttle_applied * 0.16
	for i in playback.get_frames_available():
		phase = fposmod(phase + TAU * frequency / SAMPLE_RATE, TAU)
		wind = lerpf(wind, random.randf_range(-1.0, 1.0), 0.12)
		var engine := (
			sin(phase) + 0.35 * sin(phase * 2.0) + 0.2 * sin(phase * 3.0) + 0.08 * sin(phase * 5.0)
		)
		var sample: float = engine * level + wind * clampf(sim.speed / 80.0, 0.0, 1.0) * level
		playback.push_frame(Vector2(sample, sample))
