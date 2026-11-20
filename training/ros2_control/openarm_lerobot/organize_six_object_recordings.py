"""Copy the selected six-object recordings without modifying original data.

This is an archive/inspection step, not conversion to a train-ready dataset.
No ROS imports, hardware access, motion commands, or original-file deletion.
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path


CAMERAS = ('camera1_global', 'camera2_left', 'camera3_right')
KEYS = [k for side in ('left', 'right') for k in
        [*[f'{side}_joint{i}.pos' for i in range(1, 8)], f'{side}_gripper.pos']]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def copy_checked(source, target):
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = digest(source)
    shutil.copy2(source, target)
    if digest(target) != expected:
        raise RuntimeError(f'Copy checksum mismatch: {source}')
    return {'bytes': target.stat().st_size, 'sha256': expected}


def selections(base):
    def batch(name):
        return [base / name / f'episode_{i:02d}' for i in range(1, 11)]

    def manifest(name):
        entries = read_json(base / name / 'collection_manifest.json')['episodes']
        if sorted(e['collection_episode'] for e in entries) != list(range(1, 11)):
            raise ValueError(f'Invalid collection: {name}')
        return [base / name / e['path'] for e in sorted(entries, key=lambda e: e['collection_episode'])]

    tissue = base / 'tissue_to_blue_box_demonstrations/batches'
    return [
        ('01_螺丝刀', '抓取螺丝刀', batch('screwdriver/batches/20261102_02')),
        ('02_透明胶带', '抓取透明胶带', batch('clear_tape/batches/20261102_01')),
        ('03_双面胶', '抓取双面胶', batch('white_tape/batches/double_sided_20261102_01')),
        ('04_抽纸', '抓取整包抽纸',
         [tissue / 'tissue_down50_left50_trial_20261102_02/episode_01'] +
         [tissue / 'tissue_remaining9_20261102_01' / f'episode_{i:02d}' for i in range(1, 10)]),
        ('05_黑色胶带', '抓取黑色胶带', manifest('black_tape')),
        ('06_红色胶带', '抓取红色胶带', manifest('red_tape')),
    ]


def validate_episode(path):
    meta = read_json(path / 'metadata.json')
    if not meta.get('completed') or abs(meta['duration_s'] - 20) > .001:
        raise ValueError(f'Incomplete episode: {path}')
    videos = {}
    for name in (*CAMERAS, 'three_views'):
        output = subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=nb_frames,duration,width,height,avg_frame_rate', '-of', 'json',
            str(path / f'{name}.mp4')], text=True)
        stream = json.loads(output)['streams'][0]
        if int(stream['nb_frames']) != 600 or abs(float(stream['duration']) - 20) > .001:
            raise ValueError(f'Video size/duration mismatch: {path}/{name}')
        if name in CAMERAS and (stream['width'], stream['height']) != (640, 480):
            raise ValueError(f'Unexpected camera shape: {path}/{name}')
        videos[name] = stream
    count, previous = 0, -math.inf
    with (path / 'commands_feedback.jsonl').open() as file:
        for line in file:
            row = json.loads(line)
            if not math.isfinite(row['monotonic']) or row['monotonic'] <= previous:
                raise ValueError(f'Invalid log time: {path}')
            previous = row['monotonic']
            for field in ('actual', 'command'):
                if len(row[field]) != 16 or not all(math.isfinite(x) for x in row[field]):
                    raise ValueError(f'Invalid {field}: {path}')
            count += 1
    if count < 2:
        raise ValueError(f'Missing action/feedback log: {path}')
    return meta, {'videos': videos, 'valid_log_rows': count}


def organize(base, output):
    output = output.resolve()
    stage = output.with_name(output.name + '.partial')
    if output.exists() or stage.exists():
        raise FileExistsError('Output or partial archive already exists; never overwrite it.')
    groups = selections(base)
    paths = [p.resolve() for _, _, episodes in groups for p in episodes]
    if len(paths) != 60 or len(set(paths)) != 60:
        raise ValueError('Expected 60 unique recordings')
    # Account for episodes, copied source logs and headroom before copying.
    required = sum(f.stat().st_size for p in paths for f in p.rglob('*') if f.is_file())
    if shutil.disk_usage(output.parent).free < required * 2 + 1024**3:
        raise RuntimeError('Insufficient free space')
    stage.mkdir()
    files, copied_batches, objects = {}, {}, []

    def copy(source, target):
        files[str(target.relative_to(stage))] = copy_checked(source, target)

    for folder, task, episodes in groups:
        entries = []
        for number, source in enumerate(episodes, 1):
            meta, checks = validate_episode(source)
            batch = Path(meta.get('source_batch', source.parent)).resolve()
            source_episode = meta.get('source_episode', meta['episode'])
            source_key = (folder, str(batch))
            evidence = Path('_来源记录') / folder / batch.name
            if source_key not in copied_batches:
                required_files = [batch / 'result.json', batch / 'commands_feedback.jsonl'] + [
                    batch / f'{camera}_timestamps.jsonl' for camera in CAMERAS]
                for file in required_files:
                    if not file.is_file():
                        raise FileNotFoundError(file)
                extra = list(batch.glob('*.json'))
                for file in sorted(set(required_files + extra)):
                    copy(file, stage / evidence / file.name)
                # Include collection-level decisions, including red/black replacements.
                for ancestor in (source.parent, source.parent.parent.parent):
                    for name in ('collection_manifest.json', 'quality_overrides.json', 'review.json'):
                        file = ancestor / name
                        target = stage / evidence / ('selection_' + ancestor.name + '_' + name)
                        if file.is_file() and not target.exists():
                            copy(file, target)
                copied_batches[source_key] = str(evidence)
            # Verify clip offsets refer to the correct source recording, not the replaced batch.
            for camera in CAMERAS:
                stamps = [json.loads(line)['monotonic'] for line in
                          (batch / f'{camera}_timestamps.jsonl').read_text().splitlines()]
                offset = meta['camera_frame_offsets'][camera]
                start = offset['start_frame']
                if start < 0 or start + 600 > len(stamps):
                    raise ValueError(f'Invalid source frame range: {source}')
                if abs(stamps[start] - offset['first_monotonic']) > .000001:
                    raise ValueError(f'Wrong source timestamps: {source}')
                if abs(stamps[start + 599] - offset['last_monotonic']) > .000001:
                    raise ValueError(f'Wrong final frame timestamp: {source}')
            target = stage / folder / f'episode_{number:02d}'
            for file in sorted(source.rglob('*')):
                if file.is_file():
                    copy(file, target / file.relative_to(source))
            quality = 'retained_recording_success_not_fully_verified'
            note = '保留录像；文件完整不等于抓取成功，正式训练前需质量复核。'
            hold = False
            if folder == '04_抽纸' and number == 1:
                quality, note = 'operator_confirmed_success', '一次成功验证，用户已确认。'
            if folder == '04_抽纸' and number == 3:
                quality, hold = 'operator_confirmed_success', False
                note = '来源为 remaining9 的 episode_02。提供录像后，用户于 2026-11-02 明确要求标记为成功；取消此前暂缓，依据为用户确认。'
            if folder == '06_红色胶带' and number == 2:
                quality = 'sampled_visual_success_operator_confirmation_pending'
                note = '新补录；抽查可见红色胶带抓起并放入盒内，替代用户拒绝的旧第二段。'
            entry = {'episode': number, 'task': task, 'directory': str(target.relative_to(stage)),
                     'original_directory': str(source.resolve()), 'source_batch': str(batch),
                     'source_episode': source_episode, 'source_evidence_directory': str(evidence),
                     'quality_status': quality, 'quality_note': note,
                     'hold_from_success_training_until_review': hold, 'checks': checks}
            write_json(target / '整理来源.json', entry)
            entries.append(entry)
            print(f'{folder} {number}/10: checked and copied', flush=True)
        summary = {'task': task, 'episode_count': 10, 'episodes': entries}
        write_json(stage / folder / '清单.json', summary)
        objects.append(summary)
    index = {'schema_version': 1, 'stage': 'organized_raw_recordings_not_train_ready',
             'object_count': 6, 'episode_count': 60, 'duration_s_per_episode': 20,
             'frames_per_camera_per_episode': 600, 'fps': 30,
             'camera_mapping': {'camera1_global': '全局', 'camera2_left': '左腕（黄色夹爪）',
                                'camera3_right': '右腕（白色夹爪）'},
             'joint_order': KEYS, 'command_field': 'command', 'observation_field': 'actual',
             'units': 'arm: radians; gripper: normalized by 0.03 m; actual feedback retained without clipping',
             'hardware_synchronized': False, 'originals_modified': False,
             'excluded': ['旧红色胶带第2段', '各物品 rejected 旧片段', '早期失败抽纸试验',
                          '重复示教预览及原始示教（与本次批量合集分开保留）'],
             'objects': objects}
    write_json(stage / '总清单.json', index)
    write_json(stage / '文件校验_SHA256.json', files)
    instructions = '''六类物品抓取数据（整理后的原始录像，尚非可直接训练的 LeRobot 数据集）

六个物品目录各 10 段，共 60 段；每段 20 秒。
camera1_global.mp4：全局；camera2_left.mp4：左腕；camera3_right.mp4：右腕。
three_views.mp4：三路拼接，仅供观看，不是额外一次抓取。
commands_feedback.jsonl：command 为发送动作；actual 为实测关节反馈，16 维。
metadata.json：原始元数据，保持不变；整理来源.json：本合集编号、真实源批次及质量备注。
_来源记录：每个源批次的时间戳、全量动作日志、执行结果和可用的保留清单。
总清单.json：完整索引；文件校验_SHA256.json：复制文件的 SHA-256，已与源文件核对。

注意：本目录是独立复制，不是指向原目录的视频快捷方式；原始文件未删除、未改动。
红色胶带第 2 段使用最新补录，旧第 2 段未收入本合集；其他替换记录已按来源对应。
抽纸第 1 段是成功验证，后面 9 段来自 remaining9 批次。
抽纸第 3 段（原 remaining9/episode_02）已于 2026-11-02 由用户确认成功，后续转换默认纳入。
其余录像也不因复制成功就自动认定抓取成功。60 段是录制数量，不是已验证成功数量。
部分执行结果含最终返回弯举未到位错误；保留记录，不改写为整次执行成功。
视频来自同一示教轨迹的反复回放，不等于 60 条不同轨迹；不能据此宣称新位置泛化。
后续训练前仍需按源批次时间戳对齐图像/状态/动作、复核成功标签并转换为 LeRobot 格式。
此整理工具不连接机械臂，不控制电机，不启动训练。
'''
    with (stage / '使用说明.txt').open('x', encoding='utf-8') as file:
        file.write(instructions)
    stage.rename(output)
    print(f'COMPLETE: {output}; 60 episodes; {len(files)} source files verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent / 'datasets')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    organize(args.source.resolve(), args.output)
