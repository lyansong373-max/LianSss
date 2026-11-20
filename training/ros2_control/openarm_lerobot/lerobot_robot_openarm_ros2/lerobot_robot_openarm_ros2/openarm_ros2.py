"""
双臂 OpenArm v10（ROS2 控制）的 LeRobot Robot 插件。

硬件由 ROS2 ros2_control 控制，本类通过 UDP 与 ROS2 端的 openarm_lerobot_bridge.py 通信：
  get_observation(): 关节状态（弧度）+ 夹爪（0~1）+ 相机图像
  send_action():     目标关节（弧度）+ 夹爪（0~1）→ 桥接节点做限位/限幅后下发
"""

import json
import logging
import socket
import threading
import time
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot

from .config_openarm_ros2 import OpenArmROS2Config

logger = logging.getLogger(__name__)

SIDES = ("left", "right")
# 顺序：left_joint1..7, left_gripper, right_joint1..7, right_gripper
MOTORS = [m for s in SIDES for m in [f"{s}_joint{i}" for i in range(1, 8)] + [f"{s}_gripper"]]


class OpenArmROS2(Robot):
    config_class = OpenArmROS2Config
    name = "openarm_ros2"

    def __init__(self, config: OpenArmROS2Config):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._dummy_cams = [f"camera{i}" for i in range(1, config.dummy_cameras + 1)]
        overlap = set(self._dummy_cams) & set(self.cameras)
        if overlap:
            raise ValueError(f"dummy 相机名与真实相机重名: {overlap}")

        self._sock: socket.socket | None = None
        self._rx_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._state: dict[str, float] | None = None
        self._state_t = 0.0

    # ------------------------------------------------------------------ features
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in MOTORS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        ft = {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}
        for k in self._dummy_cams:
            ft[k] = (self.config.dummy_camera_height, self.config.dummy_camera_width, 3)
        return ft

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ------------------------------------------------------------------ connection
    @property
    def is_connected(self) -> bool:
        return self._running and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", self.config.state_port))
        self._sock.settimeout(0.2)
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

        deadline = time.monotonic() + self.config.connect_timeout_s
        while self._state is None:
            if time.monotonic() > deadline:
                self.disconnect()
                raise ConnectionError(
                    f"{self.config.connect_timeout_s}s 内没收到 OpenArm 状态。请确认：\n"
                    "  1) 硬件控制 launch（openarm.bimanual.launch.py）已启动且 /joint_states 有数据\n"
                    "  2) openarm_lerobot_bridge.py 正在运行，state_port 一致"
                )
            time.sleep(0.05)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # 零位与限位由 ROS2/电机端负责，这里无需标定
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self._running = False
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------ IO
    def _rx_loop(self) -> None:
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                state = json.loads(data.decode())["state"]
            except (ValueError, KeyError):
                continue
            with self._lock:
                self._state = state
                self._state_t = time.monotonic()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        with self._lock:
            state, age = self._state, time.monotonic() - self._state_t
        if age > self.config.max_state_age_s:
            raise RuntimeError(f"OpenArm 状态已 {age:.2f}s 未更新（桥接节点或 ROS2 是否挂了？）")

        obs = {f"{m}.pos": float(state[f"{m}.pos"]) for m in MOTORS}
        for key, cam in self.cameras.items():
            obs[key] = cam.async_read()
        for key in self._dummy_cams:
            obs[key] = np.zeros(
                (self.config.dummy_camera_height, self.config.dummy_camera_width, 3), dtype=np.uint8
            )
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        goal = {f"{m}.pos": float(action[f"{m}.pos"]) for m in MOTORS}
        payload = json.dumps({"t": time.time(), "action": goal}).encode()
        self._sock.sendto(payload, (self.config.bridge_host, self.config.cmd_port))
        # 真正的限位/单步限幅在桥接节点里做，这里返回请求值
        return goal
