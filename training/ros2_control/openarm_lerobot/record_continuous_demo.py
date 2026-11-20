#!/usr/bin/env python3
"""Auto-start: wait 15s, control without recording for 5s, then record.

No positional alignment gate or homing. Retain freshness/validity checks;
pass targets through as in the original teleoperation, with ROS bridge limits
unchanged. Right arrow in the recording terminal ends and saves an episode.
Only this process wraps LeRobot; installed LeRobot files remain unchanged.
"""
import inspect
import math
import socket
import sys
import time

from start_pose import positions, register_local_plugins

TAKEOVER_DELAY_S = 15.0
PREROLL_S = 5.0


def handle_demo_key(key, events):
    # Terminal-only controls. Space never starts, stops, or pauses anything.
    if key in ("esc", "q"):
        events["stop_recording"] = True
        events["exit_early"] = True
    elif events.get("demo_phase") == "recording":
        if key in ("right", "n"):
            events["exit_early"] = True
        elif key in ("left", "r"):
            events["exit_early"] = True
            events["rerecord_episode"] = True


def init_demo_keyboard():
    from lerobot.utils.keyboard_input import TerminalKeyListener
    if not sys.stdin.isatty():
        raise RuntimeError("请在现场交互终端运行，以保留备用退出键")
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False,
              "demo_phase": "preparing"}
    listener = TerminalKeyListener(lambda key: handle_demo_key(key, events))
    listener.start()
    print("自动开始；空格无作用。仅在本终端：录制时右箭头→结束并保存（n备用），←/r重录，Esc/q退出。"
          "这些按键不是硬件急停。", flush=True)
    return listener, events


def check_cancel(events):
    if events.get("stop_recording", False):
        raise KeyboardInterrupt


def countdown_to_control(robot, teleop, events, *, duration_s=TAKEOVER_DELAY_S,
                         clock=time.monotonic, sleep=time.sleep):
    """Read fresh data only; never send actions or compare pose errors."""
    check_cancel(events)
    events["demo_phase"] = "countdown"
    events["exit_early"] = False
    events["rerecord_episode"] = False
    start = clock()
    last_count = None
    print("设备就绪：15秒后自动接管；期间不发送动作。请大致摆好姿态，确认周围安全。", flush=True)
    print("连续示范已取消夹爪实际反馈范围检查；保留缺失/非有限值/过期检查。"
          "目标仍为[0, 1]，保留原始反馈，不改变速度或行程。", flush=True)
    while True:
        check_cancel(events)
        current = positions(robot.get_observation(), check_gripper_range=False)
        positions(teleop.get_action())
        check_cancel(events)
        left = max(0, math.ceil(duration_s - (clock() - start)))
        if left != last_count:
            print(f"接管倒计时：{left} 秒", flush=True)
            last_count = left
        if clock() - start >= duration_s:
            return current
        sleep(1 / 30)


class DemoActionProcessor:
    def __init__(self, original, events, *, clock=time.monotonic):
        self.original = original
        self.events = events
        self.clock = clock
        self.warmup_start = clock()
        self.last_count = None
        self.recording = False

    def __call__(self, action_observation):
        check_cancel(self.events)
        _, observation = action_observation
        positions(observation, check_gripper_range=False)
        requested = positions(self.original(action_observation))
        now = self.clock()
        if not self.recording:
            left = max(0, math.ceil(PREROLL_S - (now - self.warmup_start)))
            if left != self.last_count:
                print(f"已接管，尚未录制：{left}秒后开始录制。", flush=True)
                self.last_count = left
        check_cancel(self.events)
        return requested


def prepare_demo_loop(original):
    signature = inspect.signature(original)

    def record_loop(*args, **kwargs):
        if "robot" in signature.parameters:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = dict(bound.arguments)
        else:
            if args:
                raise TypeError("当前 LeRobot 录制适配器要求 record_loop 使用关键字参数")
            values = dict(kwargs)
        if values.get("dataset") is None:
            return original(*args, **kwargs)
        robot, teleop = values["robot"], values.get("teleop")
        if getattr(robot, "name", None) != "openarm_ros2" or getattr(teleop, "name", None) != "openarm_exo":
            raise ValueError("连续示范要求 openarm_ros2 机器人和 openarm_exo 遥操作设备")
        events = values["events"]
        countdown_to_control(robot, teleop, events)
        check_cancel(events)
        processor = DemoActionProcessor(values['teleop_action_processor'], events)
        values['teleop_action_processor'] = processor
        events['demo_phase'] = 'preroll'
        print("开始自动接管：恢复原始目标跟随，不额外限速；5秒后录制。"
              "ROS桥接原有限位/单步限幅仍有效；姿态不一致可能快速追赶目标，不做碰撞规划。", flush=True)
        warmup = {**values, 'dataset': None, 'control_time_s': PREROLL_S, 'timer': None}
        original(**warmup)
        check_cancel(events)
        if events.get('exit_early') or events.get('rerecord_episode'):
            raise RuntimeError("准备阶段被中断，本次不开始录制")
        processor.recording = True
        events['demo_phase'] = 'recording'
        timer = values.get('timer')
        if timer is not None:
            timer.restart()
        print("正式开始录制！可先保持弯举2秒，再执行当前任务的完整示范。", flush=True)
        print("完成后聚焦本录制终端，按右箭头→结束并保存（n备用）；夹爪持续闭合不再结束录制。"
              "最长录制时限仍有效，按键不是硬件急停。", flush=True)
        try:
            result = original(**values)
            print("本条录制循环已结束，后续保存或重录由录制器处理；不自动回位、不自动判定成功。", flush=True)
            return result
        finally:
            events['demo_phase'] = 'saving'
    return record_loop


def main():
    if "--help" not in sys.argv and "-h" not in sys.argv:
        if not sys.stdin.isatty():
            raise SystemExit("请在现场交互终端运行连续示范，以保留备用退出键")
        for port in (50010, 50012):
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                try:
                    probe.bind(("0.0.0.0", port))
                except OSError as error:
                    raise SystemExit(f"UDP {port} 已被占用；先关闭其他 LeRobot 遥操作/录制/推理进程：{error}") from error
    register_local_plugins()
    from lerobot.scripts import lerobot_record
    original = lerobot_record.record_loop
    original_keyboard = lerobot_record.init_keyboard_listener
    listeners = []
    def keyboard_factory():
        listener, events = init_demo_keyboard()
        listeners.append(listener)
        return listener, events
    lerobot_record.record_loop = prepare_demo_loop(original)
    lerobot_record.init_keyboard_listener = keyboard_factory
    try:
        lerobot_record.main()
    finally:
        lerobot_record.record_loop = original
        lerobot_record.init_keyboard_listener = original_keyboard
        for listener in listeners:
            listener.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止；中断的示范不能视为完整成功轨迹。")
        raise SystemExit(130)
