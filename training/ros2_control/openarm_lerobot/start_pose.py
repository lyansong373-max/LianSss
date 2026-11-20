#!/usr/bin/env python3
"""Teach an actual OpenArm pose, then return to it without recording reset motion.

No joint angles are supplied by default. A pose must first be demonstrated on
the real robot. Paths between poses are joint-space interpolations, not
collision-checked trajectories; the operator must keep the return path clear.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import select
import socket
import sys
import tempfile
import time

SIDES = ("left", "right")
ARM_KEYS = tuple(f"{side}_joint{i}.pos" for side in SIDES for i in range(1, 8))
GRIPPER_KEYS = tuple(f"{side}_gripper.pos" for side in SIDES)
# Confirmed empty, closed grippers: 2026-11-01 five-second feedback sample
# reached -0.02645 on the right at bridge scale 0.01. Observation-only margin;
# never apply this allowance to commands or saved poses, and retain raw values.
CLOSED_GRIPPER_FEEDBACK_TOLERANCE = 0.03
# User-approved observation-only opening margin (2026-11-02).
# At bridge scale 0.03, 1.25 corresponds to 0.0375 m model joint position.
# Not a velocity limit, a command allowance, or proof that tracking is safe.
OPEN_GRIPPER_FEEDBACK_TOLERANCE = 0.25
KEYS = tuple(key for side in SIDES for key in
             (*[f"{side}_joint{i}.pos" for i in range(1, 8)], f"{side}_gripper.pos"))


def register_local_plugins() -> None:
    # Executing a file in openarm_lerobot puts the outer editable-project
    # directories on sys.path. Add their actual package roots explicitly to
    # avoid importing empty namespace packages with the same names.
    root = Path(__file__).resolve().parent
    for name in ("lerobot_robot_openarm_ros2", "lerobot_teleoperator_openarm_exo"):
        project = root / name
        if (project / name / "__init__.py").is_file() and str(project) not in sys.path:
            sys.path.insert(0, str(project))


@dataclass(frozen=True)
class ReturnSettings:
    minimum_duration_s: float = 5.0
    max_velocity_rad_s: float = 0.25
    joint_tolerance_rad: float = 0.06
    gripper_tolerance: float = 0.08
    max_following_error_rad: float = 0.30
    settle_s: float = 0.8
    settle_timeout_s: float = 15.0
    handover_timeout_s: float = 90.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")


def positions(values: dict, *, allow_closed_gripper_feedback: bool = False,
              allow_open_gripper_feedback: bool = False,
              check_gripper_range: bool = True) -> dict[str, float]:
    """Validate 16 channels; only raw observations may opt out of grip range checks."""
    missing = set(KEYS) - set(values)
    if missing:
        raise ValueError(f"缺少状态/动作通道：{sorted(missing)}")
    result = {key: float(values[key]) for key in KEYS}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("状态/动作包含 NaN 或 Inf")
    if not check_gripper_range:
        return result
    lower = -CLOSED_GRIPPER_FEEDBACK_TOLERANCE if allow_closed_gripper_feedback else 0.0
    upper = 1.0 + OPEN_GRIPPER_FEEDBACK_TOLERANCE if allow_open_gripper_feedback else 1.0
    invalid = {key: result[key] for key in GRIPPER_KEYS if not lower <= result[key] <= upper}
    if invalid:
        details = ", ".join(f"{key}={value:.6f}" for key, value in invalid.items())
        raise ValueError(f"夹爪归一化值超出允许范围 [{lower:g}, {upper:g}]：{details}；"
                         "请检查夹爪状态换算及开合位置")
    return result


def load_pose(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("robot_type") != "openarm_ros2":
        raise ValueError("起始姿态文件格式或机器人类型不匹配")
    return positions(data["positions"])


def save_pose(path: Path, pose: dict[str, float]) -> None:
    pose = positions(pose)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the previous taught pose recoverable when teaching a replacement.
    if path.exists():
        backup = path.with_name(path.name + "." + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
        backup.write_bytes(path.read_bytes())
        print(f"上一次姿态已备份：{backup}", flush=True)
    payload = {
        "schema_version": 1,
        "robot_type": "openarm_ros2",
        "label": "bimanual_curl_start",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "positions": pose,
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".start_pose_", delete=False) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def pose_errors(current: dict, target: dict) -> tuple[float, float]:
    return (max(abs(current[key] - target[key]) for key in ARM_KEYS),
            max(abs(current[key] - target[key]) for key in GRIPPER_KEYS))


def close_to_pose(current: dict, target: dict, settings: ReturnSettings) -> bool:
    joint_error, gripper_error = pose_errors(current, target)
    return joint_error <= settings.joint_tolerance_rad and gripper_error <= settings.gripper_tolerance


def motion_duration(start: dict, target: dict, settings: ReturnSettings) -> float:
    # smoothstep has a maximum derivative of 1.5. This bounds requested speed
    # per joint independently of the distance to the taught pose.
    joint_distance, _ = pose_errors(start, target)
    return max(settings.minimum_duration_s, 1.5 * joint_distance / settings.max_velocity_rad_s)


def interpolated_pose(start: dict, target: dict, fraction: float) -> dict[str, float]:
    fraction = min(1.0, max(0.0, fraction))
    weight = fraction * fraction * (3.0 - 2.0 * fraction)
    return {key: start[key] + (target[key] - start[key]) * weight for key in KEYS}


def move_to_start(robot, target: dict, *, teleop=None, fps: float = 30,
                  settings: ReturnSettings | None = None, stopped=lambda: False,
                  clock=time.monotonic, sleep=time.sleep) -> None:
    """Return to a taught pose and optionally wait for exoskeleton alignment.

    No frames are written here. Invalid/stale observations, stale teleop input,
    excessive tracking error and cancellation abort before any new target is
    sent. The low-level ROS controller retains its last target on exit.
    """
    settings = settings or ReturnSettings()
    target = positions(target)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须是有限正数")

    def check_cancel():
        if stopped():
            raise KeyboardInterrupt("起始姿态准备已取消")

    check_cancel()
    start = positions(robot.get_observation())
    if teleop is not None:
        positions(teleop.get_action())  # Check freshness before starting motion.
    duration = motion_duration(start, target, settings)
    print(f"自动回到弯举起始姿态，预计 {duration:.1f} 秒；此过程不录制。", flush=True)
    began = clock()
    previous_command = start
    stable_since = None
    while True:
        tick = clock()
        check_cancel()
        current = positions(robot.get_observation())
        if teleop is not None:
            positions(teleop.get_action())
        following_error, _ = pose_errors(current, previous_command)
        if following_error > settings.max_following_error_rad:
            raise RuntimeError(f"回位跟随误差过大（{following_error:.3f} rad），已停止发送新目标")
        elapsed = clock() - began
        if elapsed > duration + settings.settle_timeout_s:
            raise TimeoutError("机械臂未在规定时间内到达弯举姿态；检查实际位置、限位、夹爪及 dry_run")
        command = interpolated_pose(start, target, elapsed / duration)
        check_cancel()
        robot.send_action(command)
        previous_command = command
        if elapsed >= duration and close_to_pose(current, target, settings):
            stable_since = clock() if stable_since is None else stable_since
            if clock() - stable_since >= settings.settle_s:
                break
        else:
            stable_since = None
        sleep(max(0.0, 1 / fps - (clock() - tick)))

    if teleop is None:
        print("已到弯举起始姿态。", flush=True)
        return

    print("已回位。请把外骨骼和夹爪目标对齐到同一姿态；稳定对齐后自动开始录制。", flush=True)
    began = clock()
    stable_since = None
    while clock() - began <= settings.handover_timeout_s:
        tick = clock()
        check_cancel()
        current = positions(robot.get_observation())
        requested = positions(teleop.get_action())
        if not close_to_pose(current, target, settings):
            raise RuntimeError("等待外骨骼对齐时机械臂偏离起始姿态，已中止")
        check_cancel()
        robot.send_action(target.copy())
        if close_to_pose(requested, target, settings):
            stable_since = clock() if stable_since is None else stable_since
            if clock() - stable_since >= settings.settle_s:
                print("外骨骼已对齐，开始本条示范。", flush=True)
                return
        else:
            stable_since = None
        sleep(max(0.0, 1 / fps - (clock() - tick)))
    raise TimeoutError("外骨骼未对齐起始姿态；本条没有开始录制")


def teach_pose(robot, teleop, path: Path, *, fps: float = 30) -> None:
    """Follow the exoskeleton until the operator saves a stable measured pose."""
    print("示教：用外骨骼让双臂摆成所需弯举姿态，保持稳定。", flush=True)
    print("终端输入 s 后回车保存并退出；输入 q 后回车退出。此模式会遥操作真机。", flush=True)
    history = deque()
    settings = ReturnSettings()
    while True:
        tick = time.monotonic()
        current = positions(robot.get_observation())
        requested = positions(teleop.get_action())
        robot.send_action(requested)
        history.append((tick, current))
        while history and tick - history[0][0] > 1.0:
            history.popleft()
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line or line.strip().lower() == "q":
                return
            if line.strip().lower() == "s":
                if (not history or tick - history[0][0] < 0.7
                        or any(not close_to_pose(state, current, settings) for _, state in history)
                        or not close_to_pose(requested, current, settings)):
                    print("尚未稳定或实际位置未跟上外骨骼，请保持姿态后再次输入 s。", flush=True)
                    continue
                save_pose(path, current)
                print(f"已保存实际起始姿态：{path}", flush=True)
                return
        time.sleep(max(0.0, 1 / fps - (time.monotonic() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("teach", "home", "show"))
    parser.add_argument("--pose", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "show":
        print(json.dumps(load_pose(args.pose), ensure_ascii=False, indent=2))
        return
    if not sys.stdin.isatty():
        raise SystemExit("请在交互终端中运行姿态示教/回位命令")
    target = load_pose(args.pose) if args.mode == "home" else None
    # The installed plugins use SO_REUSEADDR. Explicitly check for another
    # client first, so teaching/homing cannot silently share its UDP port.
    for port in ((50010, 50012) if args.mode == "teach" else (50010,)):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as error:
                raise SystemExit(f"无法独占 UDP {port}；请先关闭其他 LeRobot 遥操作/录制/推理进程：{error}") from error
    register_local_plugins()
    from lerobot_robot_openarm_ros2 import OpenArmROS2, OpenArmROS2Config
    from lerobot_teleoperator_openarm_exo import OpenArmExo, OpenArmExoConfig

    robot = OpenArmROS2(OpenArmROS2Config(id="openarm_bi"))
    teleop = OpenArmExo(OpenArmExoConfig(id="qnbot_exo", fail_on_stale=True)) if args.mode == "teach" else None
    try:
        if teleop is not None:
            teleop.connect()
        robot.connect()
        if args.mode == "teach":
            teach_pose(robot, teleop, args.pose)
        else:
            move_to_start(robot, target)
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作已停止，不再发送新的动作目标。", flush=True)
        raise SystemExit(130)
