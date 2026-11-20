#!/usr/bin/env python3
"""Run installed lerobot-record with an unrecorded start-pose preparation phase.

The hook is scoped to this process; installed LeRobot sources are not modified.
Only loops with a dataset (actual recording, including rerecords) are prepared.
The normal unrecorded reset loop remains available for putting objects back.
"""

import argparse
import inspect
from pathlib import Path
import sys

from start_pose import load_pose, move_to_start, register_local_plugins


def prepare_record_loop(original, pose):
    signature = inspect.signature(original)

    def record_loop(*args, **kwargs):
        if "robot" in signature.parameters:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
        else:
            # LeRobot 0.6.2's safe_stop_image_writer decorator does not retain
            # the inner signature. Its recorder calls both phases by keyword.
            if args:
                raise TypeError("该 LeRobot record_loop 装饰器只支持关键字调用")
            values = kwargs
        if values.get("dataset") is not None:
            robot = values["robot"]
            teleop = values.get("teleop")
            if getattr(robot, "name", None) != "openarm_ros2":
                raise ValueError("弯举起始姿态只适用于 openarm_ros2")
            if teleop is None or isinstance(teleop, list):
                raise ValueError("自动回位采集要求单个 openarm_exo 遥操作设备")
            events = values["events"]
            move_to_start(robot, pose, teleop=teleop, fps=values["fps"],
                          stopped=lambda: events.get("stop_recording", False))
            # An arrow pressed during preparation must not discard the first
            # recording tick or accidentally save an empty episode.
            events["exit_early"] = False
            timer = values.get("timer")
            if timer is not None:
                timer.restart()
        return original(*args, **kwargs)

    return record_loop


def main():
    if "--" not in sys.argv:
        raise SystemExit("用法：record_with_start_pose.py --pose FILE -- <lerobot-record 参数>")
    separator = sys.argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:separator])
    pose = load_pose(args.pose)
    forwarded = sys.argv[separator + 1:]
    register_local_plugins()
    from lerobot.scripts import lerobot_record

    sys.argv = ["lerobot-record", *forwarded]
    original = lerobot_record.record_loop
    lerobot_record.record_loop = prepare_record_loop(original, pose)
    try:
        lerobot_record.main()
    finally:
        lerobot_record.record_loop = original


if __name__ == "__main__":
    main()
