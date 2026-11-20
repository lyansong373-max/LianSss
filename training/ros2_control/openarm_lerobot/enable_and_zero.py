#!/usr/bin/env python3
"""One-terminal enable -> readiness checks -> slow zero -> keep driver alive.

Default/help only describes the workflow. --execute requires interactive ZERO
confirmation before starting anything. Never rewrites motor encoder zeros.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from slow_demo_replay import CAMERAS, NAMES, ZERO_GRIPPER_TOLERANCE

ROOT = Path(__file__).resolve().parent
DDS_PROFILE = ROOT / 'config' / 'fastdds_udp.xml'
CONTROLLER_QUERY_INTERVAL_S = 0.5
ROS2 = '/opt/ros/humble/bin/ros2'
TOPICS = [f'/{side}_forward_position_controller/commands' for side in ('left', 'right')]
CONTROLLERS = {'joint_state_broadcaster', 'left_forward_position_controller',
               'right_forward_position_controller', 'left_gripper_controller',
               'right_gripper_controller'}
DRIVER = [ROS2, 'launch', 'openarm_bringup', 'openarm.bimanual.launch.py',
          'arm_type:=v10', 'use_fake_hardware:=false',
          'robot_controller:=forward_position_controller',
          'right_can_interface:=can0', 'left_can_interface:=can1', 'launch_rviz:=false']
PLANNER = [ROS2, 'launch', 'openarm_bimanual_moveit_config', 'move_group.launch.py',
           'allow_trajectory_execution:=false']
BLOCKED_NAMES = {
    'ros2_control_node', 'openarm.bimanual.launch.py', 'openarm.launch.py',
    'openarm_lerobot_bridge.py', 'exoskeleton_bridge', 'exoskeleton_bridge_node',
    'exoskeleton_bridge.launch.py', 'record_continuous_demo.py',
    'record_with_start_pose.py', 'slow_demo_replay.py', 'lerobot-record',
    'lerobot-teleoperate', 'lerobot-rollout', 'move_group', 'move_group.launch.py',
    'set_zero.sh', 'motor-check', 'openarm-demo', 'enable_and_zero.py',
}


def configure_transport():
    """Before rclpy.init; inherited only by this supervisor's children.

    Avoid the shared-memory transport on this startup path without deleting
    global SHM files or changing shell/daemon configuration. Never silently
    replace a user-selected middleware or XML configuration.
    """
    if os.environ.get('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp') != 'rmw_fastrtps_cpp':
        raise RuntimeError('此入口使用 rmw_fastrtps_cpp UDP 配置；请在新终端核对 RMW_IMPLEMENTATION')
    for key in ('FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE'):
        value = os.environ.get(key)
        if value and Path(value).resolve() != DDS_PROFILE.resolve():
            raise RuntimeError(f'{key} 已指定其他配置，拒绝覆盖：{value}')
    if not DDS_PROFILE.is_file():
        raise RuntimeError(f'缺少启动通信配置：{DDS_PROFILE}')
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'
    os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(DDS_PROFILE)
    print('启动通信使用 Fast DDS UDP-only；只影响本程序及子进程，不清理共享内存。', flush=True)


def conflicting_processes(proc_root=Path('/proc')):
    conflicts = []
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal() or int(entry.name) in (os.getpid(), os.getppid()):
            continue
        try:
            arguments = entry.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError(f'无法检查进程 {entry.name}，拒绝自动启动') from exc
        tokens = {Path(arg.decode(errors='replace')).name for arg in arguments if arg}
        found = sorted(tokens & BLOCKED_NAMES)
        if found:
            conflicts.append((int(entry.name), ', '.join(found)))
    return conflicts


def require_no_conflicts():
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError('已有驱动/运动程序/规划服务，请先核对并在其终端正常结束；不会替你关闭：\n' +
                           '\n'.join(f'  PID {pid}: {name}' for pid, name in conflicts))


def require_files():
    paths = [Path(ROS2), ROOT / 'slow_demo_replay.py',
             ROOT.parent / 'install/openarm_hardware/lib/libopenarm_hardware.so']
    paths += [Path(f'/dev/v4l/by-path/pci-0000:00:14.0-usb-0:{port}:1.0-video-index0')
              for port in CAMERAS.values()]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError('缺少程序或相机设备：\n' + '\n'.join(missing))


def fresh_feedback(state, now):
    return all(name in state and math.isfinite(state[name][0]) and
               0 <= now - state[name][1] <= 0.3 for name in NAMES)


def verify_zero_result(output):
    result = json.loads((output / 'result.json').read_text())
    pose = json.loads((output / 'zero_pose.json').read_text())
    if result.get('status') != 'zero_reached_and_saved' or pose.get('encoder_zero_rewritten') is not False:
        raise RuntimeError('回零进程未留下成功验收记录')
    from slow_demo_replay import KEYS
    positions = pose.get('positions', {})
    measured = pose.get('measured_positions', {})
    if set(positions) != set(KEYS) or any(positions[key] != 0 for key in KEYS):
        raise RuntimeError('保存的目标不是标准机械零位')
    for key in KEYS:
        value = measured.get(key)
        tolerance = ZERO_GRIPPER_TOLERANCE if 'gripper' in key else 0.02
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > tolerance:
            raise RuntimeError(f'零位实测验收不合格：{key}={value}')


class Children:
    """Only signal process groups created by this instance, never pgrep/killall."""
    def __init__(self, output):
        self.output = output
        self.items = {}
        self.log_offsets = {}
        self.log_fragments = {}
        self.failures = {}

    def start(self, name, command):
        if name in self.items:
            raise RuntimeError(f'拒绝重复启动 {name}')
        log = (self.output / f'{name}.log').open('x')
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        self.items[name] = (process, log)
        return process

    def alive(self, name):
        return name in self.items and self.items[name][0].poll() is None

    def require_alive(self, name):
        if not self.alive(name):
            code = self.items[name][0].poll() if name in self.items else '未启动'
            raise RuntimeError(f'{name} 已退出（{code}），见 {self.output / (name + ".log")}')
        failure = self.worker_failure(name)
        if failure:
            raise RuntimeError(f'{name} 的实际工作节点已退出，启动器仍在不代表就绪：{failure}')

    def worker_failure(self, name):
        """Launch can survive a dead worker. Inspect only our own launch log.

        Incremental reads also handle a line split across two reads. A failure
        stays latched; no auto-restart or motor-disable command is sent here.
        """
        if name in self.failures:
            return self.failures[name]
        worker = {'driver': 'ros2_control_node', 'planner': 'move_group'}.get(name)
        path = self.output / f'{name}.log'
        if not worker or not path.exists():
            return None
        with path.open('rb') as reader:
            reader.seek(self.log_offsets.get(name, 0))
            chunk = reader.read()
            self.log_offsets[name] = reader.tell()
        content = self.log_fragments.get(name, b'') + chunk
        lines = content.split(b'\n')
        self.log_fragments[name] = lines.pop()
        for raw in lines:
            line = raw.decode(errors='replace')
            if re.search(r'\[' + worker + r'-\d+\]: process has (?:died|finished)', line):
                self.failures[name] = line
                return line
        return None

    def stop(self, name):
        if name not in self.items:
            return
        process, _ = self.items[name]
        # Parent may have exited while launch children are still alive. Their
        # original session/process group remains ours, so signal it as well.
        for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
            # Even if launch exits first, TERM any remaining owned descendants.
            if sig == signal.SIGTERM:
                break
        if process.poll() is None:
            print(f'警告：{name} 尚未退出；请使用硬件急停并现场处理。', flush=True)

    def close(self):
        # Stop motion before driver shutdown. Software exit is not an E-stop.
        for name in ('zero', 'planner', 'driver'):
            self.stop(name)
        for _, log in self.items.values():
            log.close()

    def tail(self, name):
        path = self.output / f'{name}.log'
        if path.exists():
            print(f'\n{name} 最近日志：\n' + '\n'.join(path.read_text(errors='replace').splitlines()[-15:]), flush=True)


class Readiness:
    def __init__(self):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from controller_manager_msgs.srv import ListControllers
        from moveit_msgs.srv import GetStateValidity
        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node('enable_and_zero_supervisor')
        self.state = {}
        self.subscription = self.node.create_subscription(
            JointState, '/joint_states', self.receive,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.controller_type = ListControllers
        self.controllers = self.node.create_client(ListControllers, '/controller_manager/list_controllers')
        self.validity = self.node.create_client(GetStateValidity, '/check_state_validity')

    def receive(self, message):
        now = time.monotonic()
        for name, value in zip(message.name, message.position):
            self.state[name] = (value, now)

    def spin(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def require_no_commands(self):
        if any(self.node.count_publishers(topic) for topic in TOPICS):
            raise RuntimeError('存在机械臂命令发布者；先关闭遥操作/录制/回放桥接')

    def require_empty_graph(self):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            self.spin()
        self.require_no_commands()
        if self.node.count_publishers('/joint_states') or self.controllers.service_is_ready():
            raise RuntimeError('ROS 图中已有硬件/关节状态源，不允许另启驱动')
        if self.validity.service_is_ready():
            raise RuntimeError('已有模型检查服务，请先核对并结束旧服务后使用一键入口')

    def require_feedback(self):
        if self.node.count_publishers('/joint_states') != 1:
            raise RuntimeError('要求唯一的真机 /joint_states 发布者')
        if not fresh_feedback(self.state, time.monotonic()):
            raise RuntimeError('双臂16关节反馈缺失或超过0.3秒')

    def wait_driver(self, children, timeout=45):
        deadline = time.monotonic() + timeout
        pending = None
        active = set()
        next_query = 0.0
        while time.monotonic() < deadline:
            children.require_alive('driver')
            self.spin()
            self.require_no_commands()
            if self.node.count_publishers('/joint_states') > 1:
                raise RuntimeError('存在多个关节状态发布者')
            if pending is not None and pending.done():
                response = pending.result()
                active = {item.name for item in response.controller if item.state == 'active'}
                pending = None
            if CONTROLLERS <= active and self.node.count_publishers('/joint_states') == 1 and fresh_feedback(self.state, time.monotonic()):
                return
            if pending is None and time.monotonic() >= next_query and self.controllers.service_is_ready():
                pending = self.controllers.call_async(self.controller_type.Request())
                next_query = time.monotonic() + CONTROLLER_QUERY_INTERVAL_S
        raise RuntimeError('等待驱动/五个控制器/双臂反馈就绪超时；不会开始回零')

    def wait_planner(self, children, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            children.require_alive('driver')
            children.require_alive('planner')
            self.spin()
            self.require_no_commands()
            self.require_feedback()
            if self.validity.service_is_ready():
                return
        raise RuntimeError('等待模型检查服务超时；不会开始回零')

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def run_sequence(children, readiness, output):
    """Single pass. No retries, re-enables, or fallthrough after a failed gate."""
    print('1/3 启动驱动：验证反馈后使能并保持当前位置，不直接跳零。', flush=True)
    children.start('driver', DRIVER)
    readiness.wait_driver(children)
    print('2/3 双臂反馈与控制器就绪，启动仅规划的模型检查服务。', flush=True)
    children.start('planner', PLANNER)
    readiness.wait_planner(children)
    readiness.require_no_commands()
    readiness.require_feedback()
    print('3/3 开始回零程序：先模型/相机检查，再5秒倒计时，然后缓慢回零。', flush=True)
    process = children.start('zero', ['/usr/bin/python3', str(ROOT / 'slow_demo_replay.py'),
                                     'zero', '--out', str(output / 'zero'), '--execute'])
    # Stream only the zero worker's log; all three full logs remain on disk.
    with (output / 'zero.log').open() as reader:
        while True:
            text = reader.read()
            if text:
                print(text, end='', flush=True)
            children.require_alive('driver')
            children.require_alive('planner')
            if process.poll() is not None:
                print(reader.read(), end='', flush=True)
                if process.returncode != 0:
                    raise RuntimeError(f'回零失败，退出码 {process.returncode}；不会重试')
                break
            readiness.spin()
    verify_zero_result(output / 'zero')


def keep_driver(children, readiness):
    """Keep the enabling terminal alive; no motion is sent here."""
    warned_planner = False
    warned_feedback = False
    warned_driver = False
    while children.alive('driver'):
        readiness.spin()
        failure = children.worker_failure('driver')
        if failure and not warned_driver:
            print('硬件控制节点已退出！电机可能仍保持使能；不要继续录制或重启。\n'
                  f'{failure}\n先可靠支撑双臂并使用硬件急停，再 Ctrl+C 结束此启动器。', flush=True)
            warned_driver = True
        try:
            readiness.require_feedback()
        except RuntimeError as exc:
            if not warned_feedback:
                print(f'警告：{exc}。启动器还在不代表硬件仍在线；请停止后续操作并现场核查。', flush=True)
                warned_feedback = True
        else:
            warned_feedback = False
        if ('planner' in children.items and
                (not children.alive('planner') or children.worker_failure('planner')) and not warned_planner):
            print('模型检查服务已退出；不影响已完成的记录，但再次回位前需处理。', flush=True)
            warned_planner = True
    raise RuntimeError('底层驱动已退出；不能继续录制或认为机械臂仍保持姿态')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='现场输入 ZERO 后使能并自动缓慢回零')
    args = parser.parse_args(argv)
    if not args.execute:
        print('未启动任何程序。实际执行：bash /home/yundu1/ros2_ws/openarm_lerobot/enable_and_zero.sh --execute')
        print('会先要求输入 ZERO；随后使能、等待反馈和检查服务、预检、缓慢回零。完成后本终端必须保持运行。')
        return 0
    if not sys.stdin.isatty():
        raise RuntimeError('必须在现场交互终端运行；不允许后台/管道自动确认')
    require_files()
    require_no_conflicts()
    configure_transport()
    # Singleton for this orchestrator; no CAN operations are performed here.
    lock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    readiness = None
    children = None
    old_handlers = {}
    try:
        lock.bind('\0openarm.enable_and_zero')
        # Child launches have separate sessions. Handle terminal close/TERM too
        # so they are not intentionally left running after this supervisor exits.
        def request_stop(signum, frame):
            raise KeyboardInterrupt
        for sig in (signal.SIGTERM, signal.SIGHUP):
            old_handlers[sig] = signal.signal(sig, request_stop)
        readiness = Readiness()
        readiness.require_empty_graph()
        print('将使能双臂，然后自动缓慢回标准机械零位并闭合夹爪。不会重新标零。\n'
              '夹爪必须空载，回位路径清空，相机线不受拉扯，急停可用。模型检查不包含未建模托架/桌面。\n'
              '回零后保持驱动运行；Ctrl+C会关闭本程序启动的驱动，机械臂可能失去保持力。', flush=True)
        if input('现场确认，输入 ZERO 开始（其他输入取消）：').strip() != 'ZERO':
            print('已取消，没有启动硬件。')
            return 0
        require_files()
        require_no_conflicts()
        readiness.require_empty_graph()
        output = ROOT / 'poses' / ('enable_zero_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        children = Children(output)
        print(f'本次日志和零位记录目录：{output}', flush=True)
        try:
            run_sequence(children, readiness, output)
        except Exception as exc:
            children.stop('zero')
            print(f'自动流程已停止：{exc}', flush=True)
            for name in ('driver', 'planner', 'zero'):
                children.tail(name)
            if children.alive('driver'):
                print('驱动启动器仍运行，未自动重试或继续录制；这不保证硬件仍在线或已回零。异常时硬件急停。\n'
                      '为避免突然失去保持力，此处不自动关闭驱动；准备好安全支撑后 Ctrl+C 结束。', flush=True)
                keep_driver(children, readiness)
            return 1
        print(f'回零验收成功：{output / "zero/zero_pose.json"}\n'
              '请保持本终端运行！现在可在其他终端启动映射、桥接和录制。\n'
              '不要再开第二个底层驱动。Ctrl+C关闭本程序启动的驱动，可能失去保持力。', flush=True)
        keep_driver(children, readiness)
    except KeyboardInterrupt:
        print('\n收到退出请求：停止回零，再关闭本程序启动的规划服务和驱动。不是硬件急停。', flush=True)
        return 130
    finally:
        # A second Ctrl+C must not interrupt cleanup halfway through and leave
        # an owned motion worker running. Physical emergency stop remains vital.
        cleanup_handlers = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                cleanup_handlers[sig] = signal.signal(sig, signal.SIG_IGN)
            try:
                if children:
                    children.close()
            finally:
                if readiness:
                    readiness.close()
        finally:
            lock.close()
            for sig, handler in cleanup_handlers.items():
                signal.signal(sig, old_handlers.get(sig, handler))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'启动失败：{exc}', file=sys.stderr)
        sys.exit(1)
