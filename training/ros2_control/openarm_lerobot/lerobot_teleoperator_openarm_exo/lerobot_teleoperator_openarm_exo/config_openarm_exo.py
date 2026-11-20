from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarm_exo")
@dataclass
class OpenArmExoConfig(TeleoperatorConfig):
    # 接收外骨骼目标的本地端口（对应桥接节点的 teleop_port）
    teleop_port: int = 50012
    # connect() 时等待第一帧的超时（秒）
    connect_timeout_s: float = 10.0
    # 外骨骼数据超过这个时间没更新就打警告（秒）
    stale_warn_s: float = 0.3
    # True：外骨骼数据过期时直接报错中断录制；False：只警告并沿用最后一帧
    fail_on_stale: bool = False
