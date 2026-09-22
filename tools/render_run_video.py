"""Render a complete authoritative episode as an MP4, preserving the native HUD.

Rendering is playback of recorded states, never a second simulation. One second
of final state is retained, including for attempts with no accepted controls.
The source and an existing output are never overwritten.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_recording(source: Path, fps: int = 30) -> dict:
    if type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError('fps must be an integer from 1 to 120')
    with source.open() as stream:
        header = json.loads(next(stream))
        if header.get('type') != 'episode':
            raise ValueError('Recording must start with episode metadata')
        dt = header['physics_dt']
        if not math.isfinite(dt) or abs(dt - 1 / 120) > 1e-10:
            raise ValueError('Playback requires the game physics timestep of 1/120')
        initial = previous = header['initial_state']
        count = decisions = 0
        for line in stream:
            row = json.loads(line)
            kind = row.get('type')
            if kind == 'transition':
                state = row['state']
                if (row['previous_tick'] != previous['tick']
                        or row['tick'] != previous['tick'] + 1
                        or state['tick'] != row['tick']
                        or not math.isfinite(state['elapsed'])
                        or abs(state['elapsed'] - previous['elapsed'] - dt) > 1e-7):
                    raise ValueError('Recording transitions are discontinuous')
                previous = state
                count += 1
            elif kind == 'model_decision':
                decisions += 1
            elif kind not in ('snapshot', 'camera_observation', 'environment_failure'):
                raise ValueError(f'Unknown recording row: {kind}')
    duration = count * dt
    frames = math.ceil(duration * fps) + fps
    return {
        'source': str(source.resolve()), 'source_sha256': sha256(source),
        'episode_id': header.get('episode_id'), 'policy_id': header.get('policy_id'),
        'policy_display': header.get('policy_display', {}),
        'initial_tick': initial['tick'], 'final_tick': previous['tick'],
        'initial_elapsed_seconds': initial['elapsed'],
        'final_elapsed_seconds': previous['elapsed'],
        'transitions': count, 'model_decisions': decisions,
        'simulation_duration_seconds': duration, 'fps': fps,
        'expected_frames': frames, 'expected_video_duration_seconds': frames / fps,
        'final_state_hold_seconds': 1,
        'source_build': header.get('build', {}),
    }


def render_video(source: Path, output: Path, *, godot: str, ffmpeg: str,
                 fps: int = 30) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must be an MP4')
    sidecar = output.with_suffix('.mp4.json')
    work = output.with_suffix('.render')
    for path in (output, sidecar, work):
        if path.exists():
            raise FileExistsError(path)
    info = inspect_recording(source, fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir()  # Exclusive reservation; failed attempts retain diagnostic logs.
    avi = work / 'frames.avi'
    render_command = [str(godot), '--path', str(ROOT / 'godot'),
                      '--write-movie', str(avi), '--fixed-fps', str(fps),
                      '--quit-after', str(info['expected_frames']), '--',
                      '--replay=' + str(source), '--preview-camera=1']
    with (work / 'godot.log').open('x') as log:
        subprocess.run(render_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    log_text = (work / 'godot.log').read_text()
    completed = re.findall(r'REPLAY_COMPLETE ticks=(\d+)', log_text)
    if completed != [str(info['transitions'])] or 'SCRIPT ERROR:' in log_text:
        raise RuntimeError('Godot did not complete the exact recorded replay; see godot.log')
    encode_command = [str(ffmpeg), '-nostdin', '-n', '-i', str(avi), '-an',
                      '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                      '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with (work / 'ffmpeg.log').open('x') as log:
        subprocess.run(encode_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Decode every frame. This also rejects truncated/corrupt encoded output.
    verified = subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-xerror',
                               '-i', str(output), '-map', '0:v:0', '-an',
                               '-progress', 'pipe:1', '-f', 'null', '-'],
                              capture_output=True, text=True, check=True)
    progress = dict(line.split('=', 1) for line in verified.stdout.splitlines() if '=' in line)
    frames = int(progress.get('frame', -1))
    duration = int(progress.get('out_time_us', -1)) / 1_000_000
    if (frames != info['expected_frames']
            or abs(duration - info['expected_video_duration_seconds']) > 1 / fps + 1e-6):
        raise RuntimeError(f'Encoded duration/frame count mismatch: {frames}, {duration}')
    if sha256(source) != info['source_sha256']:
        raise RuntimeError('Recording changed while rendering')
    info.update({'video': str(output), 'video_sha256': sha256(output),
                 'frames': frames, 'video_duration_seconds': duration,
                 'render_command': render_command, 'encode_command': encode_command,
                 'render_game_script_sha256': sha256(ROOT / 'godot/scripts/main.gd'),
                 'render_overlay_script_sha256': sha256(ROOT / 'godot/scripts/decision_overlay.gd'),
                 'complete': True})
    with sidecar.open('x') as stream:
        json.dump(info, stream, indent=2)
        stream.write('\n')
    # Only remove the large intermediate created by this invocation after verification.
    avi.unlink()
    wav = avi.with_suffix('.wav')
    if wav.exists():
        wav.unlink()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--godot', required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(render_video(args.source, args.output, godot=args.godot,
                                  ffmpeg=args.ffmpeg, fps=args.fps), indent=2))


if __name__ == '__main__':
    main()
