from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("openarm_ros2")
@dataclass
class OpenArmROS2Config(RobotConfig):
    # openarm_lerobot_bridge 所在主机（同一台工控机就是 127.0.0.1）
    bridge_host: str = "127.0.0.1"
    # 接收机械臂状态的本地端口（对应桥接节点的 state_port）
    state_port: int = 50010
    # 发送指令到桥接节点的端口（对应桥接节点的 cmd_port）
    cmd_port: int = 50011
    # connect() 时等待第一帧状态的超时（秒）
    connect_timeout_s: float = 10.0
    # 状态超过这个时间没更新就报错（秒）
    max_state_age_s: float = 0.5
    # 相机，例如 {top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # 没有真实相机时用于跑通流程：生成 N 个全黑画面，命名为 camera1..cameraN（与 SmolVLA 期望的名字一致）
    # 仅用于测试链路，用这种数据训练出来的模型没有视觉能力
    dummy_cameras: int = 0
    dummy_camera_height: int = 240
    dummy_camera_width: int = 320
