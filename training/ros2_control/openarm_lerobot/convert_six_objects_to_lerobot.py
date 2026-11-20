"""Offline conversion of the organized OpenArm archive to LeRobot v3.

No robot access, uploads, installation or training. Source data is read-only.
Camera timestamps are host acquisition timestamps, NOT hardware exposure sync.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

CAMERAS = ('camera1_global', 'camera2_left', 'camera3_right')
KEYS = [k for s in ('left', 'right') for k in
        [*[f'{s}_joint{i}.pos' for i in range(1, 8)], f'{s}_gripper.pos']]
ENGLISH = {
    '抓取螺丝刀': 'Pick up the screwdriver and place it in the receiving box.',
    '抓取透明胶带': 'Pick up the transparent tape and place it in the receiving box.',
    '抓取双面胶': 'Pick up the double-sided tape and place it in the receiving box.',
    '抓取整包抽纸': 'Pick up the whole tissue package and place it in the receiving box.',
    '抓取黑色胶带': 'Pick up the black tape and place it in the receiving box.',
    '抓取红色胶带': 'Pick up the red tape and place it in the receiving box.',
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_lines(path: Path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside archive: {relative}')
    return path


def increasing(values, name: str):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f'Invalid timestamps: {name}')
    if np.any(np.diff(array) <= 0):
        raise ValueError(f'Non-increasing timestamps: {name}')
    return array


def nearest_indices(times, grid):
    right = np.searchsorted(times, grid).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    return np.where(np.abs(times[left] - grid) <= np.abs(times[right] - grid), left, right)


def preceding_indices(times, grid, max_age: float):
    indices = np.searchsorted(times, grid, side='right') - 1
    if np.any(indices < 0):
        raise ValueError('Missing action/feedback before first image; do not invent initial state')
    age = grid - times[indices]
    if np.any(age > max_age):
        raise ValueError(f'Stale action/feedback: {age.max():.6f}s > {max_age}s')
    return indices, age


def plan_alignment(camera_times, log_times, frames=600, fps=30,
                   max_image_error=.05, max_log_age=.08):
    # Same regular time grid for all modalities. Nearest camera frame is allowed
    # within 50 ms; logs use causal zero-order hold, never future commands.
    # Start at the latest first camera sample, avoiding extrapolation at start.
    grid = max(t[0] for t in camera_times.values()) + np.arange(frames) / fps
    indices, metrics = {}, {}
    for camera, times in camera_times.items():
        if np.max(np.diff(times)) > .1:
            raise ValueError(f'Camera frame gap >0.1s: {camera}')
        index = nearest_indices(times, grid)
        error = np.abs(times[index] - grid)
        if error.max() > max_image_error:
            raise ValueError(f'Image alignment error too large: {camera} {error.max():.6f}s')
        indices[camera] = index
        metrics[camera] = {
            'max_error_s': float(error.max()), 'mean_error_s': float(error.mean()),
            'repeated_frames': int(np.count_nonzero(np.diff(index) == 0)),
            'omitted_source_frames': int(frames - len(np.unique(index))),
            'frame_indices': index.tolist(),
        }
    log_indices, age = preceding_indices(log_times, grid, max_log_age)
    return grid, indices, log_indices, {
        'grid_start_monotonic': float(grid[0]), 'grid_end_monotonic': float(grid[-1]),
        'cameras': metrics, 'max_log_age_s': float(age.max()),
        'mean_log_age_s': float(age.mean()), 'source_log_indices': log_indices.tolist(),
    }


def audit(source: Path, include_held=False):
    manifest = read_json(source / '总清单.json')
    if manifest['joint_order'] != KEYS or manifest['fps'] != 30:
        raise ValueError('Unexpected joint order or FPS')
    checksums = read_json(source / '文件校验_SHA256.json')
    verified, batches, selected, excluded = set(), {}, [], []

    def checked(path):
        path = path.resolve()
        if path not in verified:
            relative = str(path.relative_to(source))
            expected = checksums[relative]
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Archive checksum mismatch: {relative}')
            verified.add(path)
        return path

    seen = set()
    for group in manifest['objects']:
        if group['task'] not in ENGLISH:
            raise ValueError(f'Unknown task: {group["task"]}')
        for entry in group['episodes']:
            directory = entry['directory']
            if directory in seen:
                raise ValueError(f'Duplicate episode: {directory}')
            seen.add(directory)
            if entry.get('hold_from_success_training_until_review') and not include_held:
                excluded.append({'directory': directory, 'reason': entry['quality_note']})
                continue
            path = safe_path(source, directory)
            meta = read_json(checked(path / 'metadata.json'))
            if not meta.get('completed') or abs(meta['duration_s'] - 20) > .001:
                raise ValueError(f'Incomplete recording: {directory}')
            evidence = safe_path(source, entry['source_evidence_directory'])
            if evidence not in batches:
                rows = read_lines(checked(evidence / 'commands_feedback.jsonl'))
                times = increasing([r['monotonic'] for r in rows], str(evidence))
                actual = np.asarray([r['actual'] for r in rows], dtype=np.float64)
                command = np.asarray([r['command'] for r in rows], dtype=np.float64)
                for array in (actual, command):
                    if array.shape != (len(rows), 16) or not np.isfinite(array).all():
                        raise ValueError(f'Invalid 16D state/action: {evidence}')
                if np.any(command[:, [7, 15]] < 0) or np.any(command[:, [7, 15]] > 1):
                    raise ValueError(f'Gripper commands outside [0,1]: {evidence}')
                stamps = {}
                for camera in CAMERAS:
                    items = read_lines(checked(evidence / f'{camera}_timestamps.jsonl'))
                    if [r['frame'] for r in items] != list(range(len(items))):
                        raise ValueError(f'Non-contiguous camera frame IDs: {evidence}/{camera}')
                    stamps[camera] = increasing([r['monotonic'] for r in items], camera)
                result = read_json(checked(evidence / 'result.json'))
                batches[evidence] = times, actual, command, stamps, result
            times, actual, command, stamps, result = batches[evidence]
            window = next((w for w in result['episode_windows']
                           if w['episode'] == entry['source_episode']), None)
            if window is None or not window.get('completed'):
                raise ValueError(f'Source recording window not completed: {directory}')
            if abs(window['start_s'] - meta['start_s']) > .001:
                raise ValueError(f'Wrong source batch/window: {directory}')
            camera_times = {}
            for camera in CAMERAS:
                video = checked(path / f'{camera}.mp4')
                probe = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                    'stream=nb_frames,duration,width,height,avg_frame_rate', '-of', 'json',
                    str(video)], text=True))['streams'][0]
                if (int(probe['nb_frames']) != 600 or abs(float(probe['duration']) - 20) > .001
                        or (probe['width'], probe['height']) != (640, 480)
                        or probe['avg_frame_rate'] != '30/1'):
                    raise ValueError(f'Unexpected video format: {video}')
                offset = meta['camera_frame_offsets'][camera]
                start = offset['start_frame']
                ts = stamps[camera][start:start + 600]
                if start < 0 or len(ts) != 600 or abs(ts[0] - offset['first_monotonic']) > 1e-6:
                    raise ValueError(f'Wrong source camera offset: {video}')
                if abs(ts[-1] - offset['last_monotonic']) > 1e-6:
                    raise ValueError(f'Wrong source final timestamp: {video}')
                camera_times[camera] = ts
            grid, image_indices, log_indices, alignment = plan_alignment(camera_times, times)
            selected.append({'entry': entry, 'path': path, 'image_indices': image_indices,
                             'actual': actual[log_indices].astype(np.float32),
                             'command': command[log_indices].astype(np.float32),
                             'alignment': alignment})
            print(f'CHECK {directory}: image max '
                  f'{max(m["max_error_s"] for m in alignment["cameras"].values()) * 1000:.1f} ms; '
                  f'log max {alignment["max_log_age_s"] * 1000:.1f} ms', flush=True)
    if not selected:
        raise ValueError('No episodes selected')
    print(f'CHECK PASSED: {len(selected)} episodes, {len(selected)*600} frames; '
          f'{len(excluded)} held for review; {len(verified)} source checksums verified', flush=True)
    return selected, excluded


class FrameReader:
    """Sequential decoder: supports repeated indices without random video seeks."""
    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.index, self.frame = -1, None
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f'Cannot decode {path}')

    def get(self, index):
        if index < self.index:
            raise ValueError('Non-monotonic selected video frames')
        while self.index < index:
            ok, bgr = self.cap.read()
            if not ok or bgr.shape != (480, 640, 3):
                raise RuntimeError(f'Corrupt frame {self.index + 1}: {self.path}')
            self.frame = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
            self.index += 1
        return self.frame.copy()

    def finish(self):
        self.get(599)
        if self.cap.read()[0]:
            raise ValueError(f'More than 600 frames: {self.path}')

    def close(self):
        self.cap.release()


def convert(source, output, repo_id, selected, excluded, language):
    # Heavy dependencies are not needed for --check-only.
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    stage = output.with_name(output.name + '.partial')
    if output.exists() or stage.exists():
        raise FileExistsError('Output or .partial already exists. Use a new output path; no overwrite.')
    output.parent.mkdir(parents=True, exist_ok=True)
    features = {k: {'dtype': 'float32', 'shape': (16,), 'names': KEYS}
                for k in ('observation.state', 'action')}
    features.update({f'observation.images.camera{i}': {
        'dtype': 'video', 'shape': (480, 640, 3), 'names': ['height', 'width', 'channels']}
        for i in range(1, 4)})
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=stage, fps=30, robot_type='openarm_ros2', features=features,
        use_videos=True, video_backend='pyav', streaming_encoding=True,
        rgb_encoder=RGBEncoderConfig(vcodec='h264', crf=20, preset='fast'),
        encoder_threads=2, encoder_queue_maxsize=8,
    )
    provenance = []
    try:
        for episode, sample in enumerate(selected):
            entry = sample['entry']
            task = (ENGLISH[entry['task']] if language == 'en'
                    else entry['task'] + '并放入接收盒')
            readers = {}
            try:
                for camera in CAMERAS:
                    readers[camera] = FrameReader(sample['path'] / f'{camera}.mp4')
                for frame in range(600):
                    item = {'observation.state': sample['actual'][frame].copy(),
                            'action': sample['command'][frame].copy(), 'task': task}
                    for i, camera in enumerate(CAMERAS, 1):
                        item[f'observation.images.camera{i}'] = readers[camera].get(
                            int(sample['image_indices'][camera][frame]))
                    dataset.add_frame(item)
                for reader in readers.values():
                    reader.finish()
                dataset.save_episode()
            finally:
                for reader in readers.values():
                    reader.close()
            provenance.append({'episode_index': episode, 'task': task, 'source': entry,
                               'alignment': sample['alignment'],
                               'raw_gripper_state_min': sample['actual'][:, [7, 15]].min(axis=0).tolist(),
                               'raw_gripper_state_max': sample['actual'][:, [7, 15]].max(axis=0).tolist()})
            print(f'SAVED {episode + 1}/{len(selected)}: {entry["directory"]}', flush=True)
    finally:
        dataset.finalize()
    # Reload from disk and check actual decoded images, labels, and numerical data.
    loaded = LeRobotDataset(repo_id=repo_id, root=stage, video_backend='pyav')
    if loaded.num_episodes != len(selected) or loaded.num_frames != len(selected) * 600:
        raise RuntimeError('Readback episode/frame count mismatch')
    for ep, sample in enumerate(selected):
        for t in (0, 300, 599):
            row = loaded[ep * 600 + t]
            for key, values in [('observation.state', sample['actual']), ('action', sample['command'])]:
                if not np.allclose(row[key].numpy(), values[t], rtol=0, atol=1e-6):
                    raise RuntimeError(f'Readback mismatch {ep}/{t}/{key}')
            if row['task'] != provenance[ep]['task']:
                raise RuntimeError(f'Task mismatch at {ep}/{t}')
            for i in range(1, 4):
                img = row[f'observation.images.camera{i}']
                if tuple(img.shape) != (3, 480, 640) or not np.isfinite(img.numpy()).all():
                    raise RuntimeError('Invalid decoded output image')
    report = {
        'status': 'converted_and_readback_checked', 'source_archive': str(source),
        'source_manifest_sha256': sha256(source / '总清单.json'), 'repo_id': repo_id,
        'episodes': len(selected), 'frames': len(selected) * 600, 'task_language': language,
        'excluded': excluded, 'provenance': provenance,
        'method': '30 Hz common grid; nearest camera frame; preceding action/feedback (zero-order hold).',
        'limits': {'max_image_alignment_s': .05, 'max_log_age_s': .08, 'max_camera_gap_s': .1},
        'caveats': ['Host acquisition timestamps, not synchronized exposure timestamps.',
                    'Actual state is preserved, including raw gripper feedback outside [0,1].',
                    'Conversion checks do not certify grasp success or generalization.',
                    'Repeated replay demonstrations; split by episode, not by random frame.'],
        'training_started': False, 'uploaded': False,
    }
    with (stage / 'conversion_report.json').open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    stage.rename(output)
    print(f'CONVERSION COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repo-id', default='local/openarm_six_objects')
    parser.add_argument('--task-language', choices=('en', 'zh'), default='en')
    parser.add_argument('--check-only', action='store_true', help='Read-only audit; no dataset is created')
    parser.add_argument('--include-held', action='store_true',
                        help='Only after manual success review: include episodes marked held for review')
    args = parser.parse_args()
    source = args.source.resolve()
    if not args.check_only:
        if args.output is None:
            parser.error('--output is required for conversion')
        output = args.output.resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            parser.error('Output must be separate from source archive')
        if output.exists() or output.with_name(output.name + '.partial').exists():
            parser.error('Output or .partial already exists; use a new directory')
    selected, excluded = audit(source, args.include_held)
    if not args.check_only:
        convert(source, output, args.repo_id, selected, excluded, args.task_language)


if __name__ == '__main__':
    main()
