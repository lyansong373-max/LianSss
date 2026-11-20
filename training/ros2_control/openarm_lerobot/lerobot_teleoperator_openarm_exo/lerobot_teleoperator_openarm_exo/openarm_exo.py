"""
Qnbot 外骨骼的 LeRobot Teleoperator 插件。

外骨骼数据链路（ROS2 侧）：
  websocket_teleoperator → /exo/joint_command → exo_retargeting
  → /left_arm/joint_command, /right_arm/joint_command → openarm_lerobot_bridge → UDP → 本类

get_action() 返回与 OpenArmROS2.action_features 完全相同的键：
  left_joint1.pos … left_joint7.pos, left_gripper.pos, right_joint1.pos … right_gripper.pos
"""

import json
import logging
import socket
import threading
import time
from functools import cached_property
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_openarm_exo import OpenArmExoConfig

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
# 顺序：left_joint1..7, left_gripper, right_joint1..7, right_gripper
MOTORS = [m for s in SIDES for m in [f"{s}_joint{i}" for i in range(1, 8)] + [f"{s}_gripper"]]


class OpenArmExo(Teleoperator):
    config_class = OpenArmExoConfig
    name = "openarm_exo"

    def __init__(self, config: OpenArmExoConfig):
        super().__init__(config)
        self.config = config
        self._sock: socket.socket | None = None
        self._rx_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._action: dict[str, Any] | None = None
        self._recv_t = 0.0
        self._last_warn_t = 0.0

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in MOTORS}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._running

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self.config.teleop_port))
        self._sock.settimeout(0.2)
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        deadline = time.monotonic() + self.config.connect_timeout_s
        while self._action is None:
            if time.monotonic() > deadline:
                self.disconnect()
                raise ConnectionError(
                    f"{self.config.connect_timeout_s}s 内没收到桥接节点数据，"
                    "请确认 openarm_lerobot_bridge.py 已运行且 teleop_port 一致"
                )
            time.sleep(0.05)
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # 外骨骼 → OpenArm 的映射由 exo_retargeting（retargeting_OpenArm.yaml）负责
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _rx_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                action = json.loads(data.decode())["action"]
            except (ValueError, KeyError):
                continue
            with self._lock:
                self._action = action
                self._recv_t = time.monotonic()

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")
        with self._lock:
            raw, bridge_age = self._action, time.monotonic() - self._recv_t
        if bridge_age > self.config.stale_warn_s:
            raise RuntimeError(f"桥接节点已 {bridge_age:.2f}s 没有数据")

        # 外骨骼本身的数据年龄（None 表示外骨骼还没开始发数据，此时目标=当前位置）
        for s in SIDES:
            age = raw.get(f"{s}_age")
            if age is None or age > self.config.stale_warn_s:
                msg = f"{s} 外骨骼数据{'尚未到达' if age is None else f'已 {age:.2f}s 未更新'}"
                if self.config.fail_on_stale:
                    raise RuntimeError(msg)
                now = time.monotonic()
                if now - self._last_warn_t > 2.0:
                    logger.warning(msg)
                    self._last_warn_t = now

        return {f"{m}.pos": float(raw[f"{m}.pos"]) for m in MOTORS}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self._running = False
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        logger.info(f"{self} disconnected.")
