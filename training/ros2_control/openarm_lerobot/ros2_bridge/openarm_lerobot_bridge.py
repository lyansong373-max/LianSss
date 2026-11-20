#!/usr/bin/env python3
"""
OpenArm <-> LeRobot 桥接节点（运行在 ROS2 Humble 的系统 Python 3.10 中）

LeRobot 需要 Python >= 3.12，无法直接 import rclpy，因此用本机 UDP 交换数据：

  ROS2 → LeRobot
    /joint_states                         → UDP :state_port   （机械臂实际状态，给 Robot.get_observation）
    /left_arm/joint_command, /right_arm/… → UDP :teleop_port  （外骨骼重定向后的目标，给 Teleoperator.get_action）
  LeRobot → ROS2
    UDP :cmd_port → /{side}_forward_position_controller/commands  (7 关节, 弧度)
                  → /{side}_gripper_controller/gripper_cmd         (夹爪, 0~1 归一化 × gripper_scale)

本节点取代 exoskeleton_bridge 的作用：录制/部署时 **不要** 同时运行 exoskeleton_bridge。

统一的键名（LeRobot 数据集中的特征名）：
  left_joint1.pos … left_joint7.pos, left_gripper.pos, right_joint1.pos … right_gripper.pos
  关节单位：弧度；夹爪：0（闭合）~1（张开）归一化。

用法：
  source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
  python3 openarm_lerobot_bridge.py                       # 正常运行
  python3 openarm_lerobot_bridge.py --ros-args -p dry_run:=true   # 只打印，不下发到机械臂
"""

import json
import math
from pathlib import Path
import socket
import threading
import time
import xml.etree.ElementTree as ET

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

SIDES = ("left", "right")
N_JOINTS = 7


def action_envelope(plan_path, margin_rad):
    """Conservative per-joint bounds from one validated demonstration plan."""
    if not math.isfinite(margin_rad) or not 0 <= margin_rad <= 0.5:
        raise ValueError("示教动作范围余量必须在 [0, 0.5] rad")
    data = json.loads(Path(plan_path).read_text())
    expected_keys = [key for side in SIDES for key in
                     ([f"{side}_joint{i}.pos" for i in range(1, 8)] + [f"{side}_gripper.pos"])]
    if data.get("keys") != expected_keys:
        raise ValueError("示教动作的关节顺序不匹配")
    actions = data.get("actions")
    if not isinstance(actions, list) or len(actions) < 2:
        raise ValueError("示教动作缺失")
    if any(not isinstance(row, list) or len(row) != 16 or
           not all(isinstance(v, (int, float)) and math.isfinite(v) for v in row)
           for row in actions):
        raise ValueError("示教动作格式或数值无效")
    return {side: [(min(row[offset + i] for row in actions) - margin_rad,
                    max(row[offset + i] for row in actions) + margin_rad)
                   for i in range(N_JOINTS)]
            for side, offset in (("left", 0), ("right", 8))}


def load_startup_pose(path):
    """Load only a successful curl's actual joint feedback, never a target."""
    result = json.loads(Path(path).read_text())
    pose = result.get("final_actual")
    if result.get("status") != "curl_preparation_completed" or not isinstance(pose, list) or \
            len(pose) != 16 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose):
        raise ValueError("弯举结果缺少有效的最终实测姿态")
    return pose


def check_startup_pose(actual, expected, side, arm_tolerance, gripper_tolerance):
    if len(actual) != 8 or len(expected) != 8 or not all(math.isfinite(v) for v in actual + expected):
        raise ValueError("首次动作姿态检查数据无效")
    index = max(range(7), key=lambda i: abs(actual[i] - expected[i]))
    drift = abs(actual[index] - expected[index])
    if drift > arm_tolerance:
        raise ValueError(f"首次动作前实际姿态已偏离弯举终点：{side}关节{index + 1} "
                         f"实测={actual[index]:.4f} rad，弯举终点={expected[index]:.4f} rad，"
                         f"偏移={drift:.4f} rad > {arm_tolerance:g} rad")
    grip_drift = abs(actual[7] - expected[7])
    if grip_drift > gripper_tolerance:
        raise ValueError(f"首次动作前{side}夹爪偏离弯举终点 {grip_drift:.3f} > "
                         f"{gripper_tolerance:g}（归一化）")


def bounded_gripper_target(value, tolerance):
    """Accept a small finite model overshoot, but never extend hardware travel."""
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.5:
        raise ValueError("夹爪命令容差必须在 [0, 0.5] 内")
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"夹爪命令无效：原始={value!r}") from None
    if not math.isfinite(raw):
        raise ValueError(f"夹爪命令无效：原始={raw}")
    if not -tolerance <= raw <= 1 + tolerance:
        raise ValueError(f"夹爪命令超出允许范围 [{-tolerance:g}, {1 + tolerance:g}]："
                         f"原始={raw:.4f}")
    return min(max(raw, 0.0), 1.0)


def bounded_arm_target(target, actual, previous, elapsed_s, limits,
                       max_target_error_rad, max_tracking_error_rad,
                       max_velocity_rad_s, max_step_rad):
    """Enforce joint limits and rate limits; optionally check policy deviations.

    This is a joint-space safeguard, not an obstacle or self-collision planner.
    """
    if not all(len(values) == N_JOINTS for values in (target, actual, limits)):
        raise ValueError("关节命令/反馈/限位长度不是 7")
    if previous is not None and len(previous) != N_JOINTS:
        raise ValueError("上一条关节命令长度不是 7")
    if not all(math.isfinite(v) for v in target + actual):
        raise ValueError("关节命令或反馈包含 NaN/Inf")
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError("命令时间间隔无效")
    if any(not lo <= p <= hi for p, (lo, hi) in zip(actual, limits)):
        raise ValueError("实际关节反馈超出 URDF 限位")
    if any(not lo <= p <= hi for p, (lo, hi) in zip(target, limits)):
        raise ValueError("模型目标超出 URDF 限位")
    if max_target_error_rad is not None:
        index = max(range(N_JOINTS), key=lambda i: abs(target[i] - actual[i]))
        target_error = abs(target[index] - actual[index])
        if target_error > max_target_error_rad:
            raise ValueError(f"模型目标偏离实际姿态超过 {max_target_error_rad:g} rad："
                             f"关节{index + 1} 目标={target[index]:.4f} rad，"
                             f"实测={actual[index]:.4f} rad，差={target_error:.4f} rad")
    if (max_tracking_error_rad is not None and previous is not None and
            max(abs(p - a) for p, a in zip(previous, actual)) > max_tracking_error_rad):
        raise ValueError(f"机械臂跟踪误差超过 {max_tracking_error_rad:g} rad")
    # Cap elapsed time too: a delayed packet must not buy a large catch-up step.
    step = min(max_step_rad, max_velocity_rad_s * min(elapsed_s, 0.05))
    base = actual if previous is None else previous
    return [p + min(max(t - p, -step), step) for t, p in zip(target, base)]


def arm_joint_name(side, i):
    return f"openarm_{side}_joint{i}"


def finger_joint_name(side):
    return f"openarm_{side}_finger_joint1"


class OpenArmLeRobotBridge(Node):
    def __init__(self):
        super().__init__("openarm_lerobot_bridge")

        self.declare_parameter("lerobot_host", "127.0.0.1")
        self.declare_parameter("state_port", 50010)   # 发给 LeRobot Robot
        self.declare_parameter("teleop_port", 50012)  # 发给 LeRobot Teleoperator
        self.declare_parameter("cmd_port", 50011)     # 从 LeRobot 接收指令
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("max_joint_state_age_s", 0.5)
        # 与用户实际正常使用的 exoskeleton_bridge gripper_scaling_factor:=0.03 一致。
        # 此系数同时用于目标换算与反馈归一化；不是实际夹口宽度。
        self.declare_parameter("gripper_scale", 0.03)
        self.declare_parameter("gripper_threshold", 0.005)  # 归一化单位下的最小变化量
        self.declare_parameter("gripper_max_effort", 10.0)
        # 每条指令每个关节最多相对上一条指令移动多少弧度（30Hz 下 0.15rad ≈ 4.5rad/s）
        self.declare_parameter("max_step_rad", 0.15)
        self.declare_parameter("dry_run", False)
        # Opt-in hardening for supervised policy rollout. Existing record/teleop
        # workflows retain their original behavior unless safe_mode is enabled.
        self.declare_parameter("safe_mode", False)
        # Disable only demonstration/target/tracking soft guards for a supervised
        # policy. URDF, fresh feedback, finite values, single publisher and rate
        # limits remain active under safe_mode.
        self.declare_parameter("enforce_policy_soft_limits", True)
        self.declare_parameter("gripper_command_tolerance", 0.05)
        self.declare_parameter("max_joint_velocity_rad_s", 0.25)
        self.declare_parameter("max_target_error_rad", 0.25)
        self.declare_parameter("max_tracking_error_rad", 0.15)
        self.declare_parameter("safety_plan_path", "")
        self.declare_parameter("startup_pose_path", "")
        self.declare_parameter("startup_arm_tolerance_rad", 0.05)
        self.declare_parameter("startup_gripper_tolerance", 0.15)
        self.declare_parameter("envelope_margin_rad", 0.10)

        gp = self.get_parameter
        self.host = gp("lerobot_host").value
        self.state_port = int(gp("state_port").value)
        self.teleop_port = int(gp("teleop_port").value)
        self.cmd_port = int(gp("cmd_port").value)
        self.gripper_scale = float(gp("gripper_scale").value)
        self.gripper_threshold = float(gp("gripper_threshold").value)
        self.gripper_effort = float(gp("gripper_max_effort").value)
        self.max_step = float(gp("max_step_rad").value)
        self.dry_run = bool(gp("dry_run").value)
        self.safe_mode = bool(gp("safe_mode").value)
        self.enforce_policy_soft_limits = bool(gp("enforce_policy_soft_limits").value)
        self.gripper_command_tolerance = float(gp("gripper_command_tolerance").value)
        self.max_joint_velocity = float(gp("max_joint_velocity_rad_s").value)
        self.max_target_error = float(gp("max_target_error_rad").value)
        self.max_tracking_error = float(gp("max_tracking_error_rad").value)
        safety_plan_path = str(gp("safety_plan_path").value)
        self.arm_envelope = (action_envelope(safety_plan_path, float(gp("envelope_margin_rad").value))
                             if self.safe_mode and self.enforce_policy_soft_limits and safety_plan_path
                             else None)
        startup_pose_path = str(gp("startup_pose_path").value)
        self.startup_pose = (load_startup_pose(startup_pose_path)
                             if self.safe_mode and startup_pose_path else None)
        self.startup_arm_tolerance = float(gp("startup_arm_tolerance_rad").value)
        self.startup_gripper_tolerance = float(gp("startup_gripper_tolerance").value)
        self.max_joint_state_age_s = float(gp("max_joint_state_age_s").value)
        if not math.isfinite(self.max_joint_state_age_s) or self.max_joint_state_age_s <= 0:
            raise ValueError("max_joint_state_age_s 必须是有限正数")
        if self.safe_mode and any(not math.isfinite(v) or v <= 0 for v in
                                  (self.max_step, self.max_joint_velocity,
                                   self.startup_arm_tolerance, self.startup_gripper_tolerance)):
            raise ValueError("safe_mode 的关节限速和起始姿态阈值必须是有限正数")
        if self.safe_mode and self.enforce_policy_soft_limits and any(
                not math.isfinite(v) or v <= 0 for v in
                (self.max_target_error, self.max_tracking_error)):
            raise ValueError("启用模型软阈值时，目标和跟踪误差上限必须是有限正数")
        if self.safe_mode and (not math.isfinite(self.gripper_command_tolerance) or
                               not 0 <= self.gripper_command_tolerance <= 0.5):
            raise ValueError("safe_mode 的夹爪命令容差必须在 [0, 0.5] 内")

        self.lock = threading.Lock()
        self.joint_pos = {}            # ROS 关节名 -> 位置
        self.joint_received_at = {}    # ROS 每个关节的实际接收时间
        self.exo_cmd = {s: None for s in SIDES}   # side -> (t, [7 关节], gripper_norm or None)
        self.last_cmd = {s: None for s in SIDES}  # 上一次下发的 7 关节
        self.last_cmd_at = {s: None for s in SIDES}
        self.last_gripper = {s: None for s in SIDES}
        self.safety_fault = None  # Latched until this process is restarted.
        self.limits = {}               # ROS 关节名 -> (lower, upper)

        # ---------- ROS 接口 ----------
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 50)
        for s in SIDES:
            self.create_subscription(
                JointState, f"/{s}_arm/joint_command", lambda m, s=s: self._on_exo(s, m), 10
            )
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/robot_description", self._on_description, latched)

        self.arm_pub = {
            s: self.create_publisher(Float64MultiArray, f"/{s}_forward_position_controller/commands", 10)
            for s in SIDES
        }
        self.gripper_client = {
            s: ActionClient(self, GripperCommand, f"/{s}_gripper_controller/gripper_cmd") for s in SIDES
        }

        # ---------- UDP ----------
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.rx.bind(("127.0.0.1" if self.safe_mode else "0.0.0.0", self.cmd_port))
        self.rx.settimeout(0.2)
        self._running = True
        threading.Thread(target=self._rx_loop, daemon=True).start()

        self.create_timer(1.0 / float(gp("publish_rate").value), self._publish_udp)
        self.create_timer(2.0, self._check_conflicts)

        self.get_logger().info(
            f"桥接已启动: state→{self.host}:{self.state_port}, teleop→{self.host}:{self.teleop_port}, "
            f"cmd←:{self.cmd_port}, gripper_scale={self.gripper_scale}, max_step={self.max_step}rad, "
            f"dry_run={self.dry_run}"
            f", safe_mode={self.safe_mode}, policy_soft_limits={self.enforce_policy_soft_limits}"
        )

    # ------------------------------------------------------------------ ROS 回调
    def _on_joint_states(self, msg: JointState):
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                self.joint_pos[n] = p
                self.joint_received_at[n] = time.monotonic()

    def _on_exo(self, side, msg: JointState):
        if len(msg.position) < N_JOINTS:
            return
        grip = None
        # 与 exoskeleton_bridge 相同的夹爪解析逻辑
        if len(msg.position) >= 8:
            grip = msg.position[7]
        elif f"{side}_gripper_joint" in msg.name:
            idx = msg.name.index(f"{side}_gripper_joint")
            if idx < len(msg.position):
                grip = msg.position[idx]
        with self.lock:
            self.exo_cmd[side] = (time.monotonic(), list(msg.position[:N_JOINTS]), grip)

    def _on_description(self, msg: String):
        try:
            root = ET.fromstring(msg.data)
        except ET.ParseError as e:
            self.get_logger().error(f"解析 /robot_description 失败: {e}")
            return
        limits = {}
        for j in root.iter("joint"):
            lim = j.find("limit")
            if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
                limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        missing = [arm_joint_name(s, i) for s in SIDES for i in range(1, 8)
                   if arm_joint_name(s, i) not in limits]
        if missing:
            self.get_logger().error(f"URDF 中缺少关节限位: {missing}")
            return
        with self.lock:
            self.limits = limits
        self.get_logger().info("已从 /robot_description 读取关节限位")

    # ------------------------------------------------------------------ ROS → LeRobot
    def _state_side(self, side):
        """当前实际状态（弧度 + 归一化夹爪），缺数据时返回 None。"""
        try:
            joints = [self.joint_pos[arm_joint_name(side, i)] for i in range(1, 8)]
            finger = self.joint_pos[finger_joint_name(side)]
            names = [arm_joint_name(side, i) for i in range(1, 8)] + [finger_joint_name(side)]
            if any(time.monotonic() - self.joint_received_at[name] > self.max_joint_state_age_s
                   for name in names):
                return None
        except KeyError:
            return None
        if not all(math.isfinite(v) for v in joints + [finger]):
            return None
        return joints, finger / self.gripper_scale

    def _publish_udp(self):
        now = time.monotonic()
        with self.lock:
            if self.safety_fault:
                return  # Make LeRobot's state watchdog end the rollout.
            state, teleop = {}, {}
            ready = True
            for s in SIDES:
                st = self._state_side(s)
                if st is None:
                    ready = False
                    continue
                joints, grip = st
                for i, v in enumerate(joints, 1):
                    state[f"{s}_joint{i}.pos"] = v
                state[f"{s}_gripper.pos"] = grip

                exo = self.exo_cmd[s]
                if exo is None:
                    # 外骨骼还没数据：遥操作目标 = 当前位置（机械臂保持不动）
                    t_joints, t_grip, age = joints, grip, None
                else:
                    t_joints = exo[1]
                    t_grip = exo[2] if exo[2] is not None else grip
                    age = now - exo[0]
                for i, v in enumerate(t_joints, 1):
                    teleop[f"{s}_joint{i}.pos"] = v
                teleop[f"{s}_gripper.pos"] = t_grip
                teleop[f"{s}_age"] = age
        if not ready:
            return
        t = time.time()
        self._send(self.state_port, {"t": t, "state": state})
        self._send(self.teleop_port, {"t": t, "action": teleop})

    def _send(self, port, payload):
        try:
            self.tx.sendto(json.dumps(payload).encode(), (self.host, port))
        except OSError as e:
            self.get_logger().warn(f"UDP 发送失败: {e}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------ LeRobot → ROS
    def _rx_loop(self):
        while self._running:
            try:
                data, _ = self.rx.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                action = json.loads(data.decode())["action"]
                self._apply_action(action)
            except Exception as e:  # noqa: BLE001
                if self.safe_mode:
                    with self.lock:
                        self._latch_safety(f"无效 UDP 指令: {e}")
                else:
                    self.get_logger().error(f"无效指令: {e}", throttle_duration_sec=2.0)

    def _apply_action(self, action):
        cmds, grips = {}, {}
        with self.lock:
            if self.safety_fault:
                return
            if not self.limits:
                self.get_logger().warn("尚未拿到 URDF 关节限位，拒绝下发指令", throttle_duration_sec=2.0)
                return
            if any(self._state_side(side) is None for side in SIDES):
                if self.safe_mode:
                    self._latch_safety("实际关节/夹爪状态缺失或超时")
                else:
                    self.get_logger().warn("实际关节/夹爪状态缺失或超时，拒绝下发指令",
                                           throttle_duration_sec=2.0)
                return
            if self.safe_mode and any(self.count_publishers(
                    f"/{s}_forward_position_controller/commands") != 1 for s in SIDES):
                self._latch_safety("机械臂控制话题存在竞争发布者")
                return
            if self.safe_mode and any(not all(f"{s}_joint{i}.pos" in action
                                              for i in range(1, 8)) for s in SIDES):
                self._latch_safety("模型指令缺少关节")
                return
            now = time.monotonic()
            for s in SIDES:
                keys = [f"{s}_joint{i}.pos" for i in range(1, 8)]
                if not all(k in action for k in keys):
                    continue
                target = [float(action[k]) for k in keys]
                if not all(math.isfinite(v) for v in target):
                    if self.safe_mode:
                        self._latch_safety(f"[{s}] 指令包含 NaN/Inf")
                        return
                    self.get_logger().error(f"[{s}] 指令包含 NaN/Inf，已丢弃")
                    continue

                # 起点：上一条指令；若没有则用当前实际位置
                prev = self.last_cmd[s]
                st = self._state_side(s)
                if self.safe_mode:
                    try:
                        if self.startup_pose is not None and prev is None:
                            offset = 0 if s == "left" else 8
                            check_startup_pose(st[0] + [st[1]],
                                               self.startup_pose[offset:offset + 8], s,
                                               self.startup_arm_tolerance,
                                               self.startup_gripper_tolerance)
                        if self.enforce_policy_soft_limits and self.arm_envelope is not None:
                            for joint, (value, (lo, hi)) in enumerate(
                                    zip(target, self.arm_envelope[s]), start=1):
                                if not lo <= value <= hi:
                                    distance = lo - value if value < lo else value - hi
                                    raise ValueError(
                                        "模型目标超出该物品的示教关节范围："
                                        f"关节{joint} 目标={value:.4f} rad，"
                                        f"允许=[{lo:.4f}, {hi:.4f}] rad，"
                                        f"超出={distance:.4f} rad")
                        elapsed = (now - self.last_cmd_at[s]
                                   if self.last_cmd_at[s] is not None else 0.02)
                        safe = bounded_arm_target(
                            target, st[0], prev, elapsed,
                            [self.limits[arm_joint_name(s, i)] for i in range(1, 8)],
                            self.max_target_error if self.enforce_policy_soft_limits else None,
                            self.max_tracking_error if self.enforce_policy_soft_limits else None,
                            self.max_joint_velocity, self.max_step,
                        )
                    except ValueError as exc:
                        self._latch_safety(f"[{s}] {exc}")
                        return
                    self.last_cmd[s] = safe
                    self.last_cmd_at[s] = now
                    cmds[s] = safe
                    g = action.get(f"{s}_gripper.pos")
                    try:
                        grip = bounded_gripper_target(g, self.gripper_command_tolerance)
                    except ValueError as exc:
                        self._latch_safety(f"[{s}] {exc}")
                        return
                    if grip != float(g):
                        self.get_logger().warn(
                            f"[{s}] 模型夹爪命令 {float(g):.4f} 已裁剪到 {grip:.4f}；"
                            "物理行程仍为 [0,1]",
                            throttle_duration_sec=5.0,
                        )
                    grips[s] = grip
                    continue
                if prev is None:
                    if st is None:
                        continue
                    prev = st[0]

                safe = []
                for i, (v, p) in enumerate(zip(target, prev), 1):
                    lo, hi = self.limits[arm_joint_name(s, i)]
                    v = min(max(v, lo), hi)                                  # 关节限位
                    v = min(max(v, p - self.max_step), p + self.max_step)    # 单步限幅
                    safe.append(v)
                self.last_cmd[s] = safe
                self.last_cmd_at[s] = now
                cmds[s] = safe

                g = action.get(f"{s}_gripper.pos")
                if g is not None and math.isfinite(float(g)):
                    grips[s] = min(max(float(g), 0.0), 1.0)

        if self.dry_run:
            self.get_logger().info(
                "dry_run " + " | ".join(
                    f"{s}: {[round(v, 3) for v in cmds.get(s, [])]} grip={grips.get(s)}" for s in SIDES
                ),
                throttle_duration_sec=0.5,
            )
            return

        with self.lock:
            if self.safety_fault:
                return
            for s, c in cmds.items():
                self.arm_pub[s].publish(Float64MultiArray(data=c))
            for s, g in grips.items():
                self._send_gripper(s, g)

    def _latch_safety(self, reason):
        if self.safety_fault is None:
            self.safety_fault = reason
            self.get_logger().error(
                f"安全熔断：{reason}；停止发送新目标。控制器仍可能保持上次目标，"
                "这不是硬件急停；请现场检查并关闭本次 rollout。"
            )

    def _send_gripper(self, side, norm):
        client = self.gripper_client[side]
        last = self.last_gripper[side]
        if last is not None and abs(norm - last) <= self.gripper_threshold:
            return
        if not client.server_is_ready():
            self.get_logger().warn(f"{side} 夹爪 action server 未就绪", throttle_duration_sec=5.0)
            return
        goal = GripperCommand.Goal()
        goal.command.position = norm * self.gripper_scale
        goal.command.max_effort = self.gripper_effort
        client.send_goal_async(goal)
        self.last_gripper[side] = norm

    # ------------------------------------------------------------------ 安全检查
    def _check_conflicts(self):
        for s in SIDES:
            n = self.count_publishers(f"/{s}_forward_position_controller/commands")
            if n > 1:
                if self.safe_mode:
                    with self.lock:
                        self._latch_safety(f"/{s}_forward_position_controller/commands 有 {n} 个发布者")
                self.get_logger().warn(
                    f"/{s}_forward_position_controller/commands 有 {n} 个发布者！"
                    "请关闭 exoskeleton_bridge，否则会和 LeRobot 抢着控制机械臂",
                    throttle_duration_sec=10.0,
                )
        if not self.limits:
            self.get_logger().warn("还没收到 /robot_description（硬件控制 launch 启动了吗？）",
                                   throttle_duration_sec=10.0)

    def destroy_node(self):
        self._running = False
        self.rx.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = OpenArmLeRobotBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
