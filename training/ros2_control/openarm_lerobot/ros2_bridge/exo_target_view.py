#!/usr/bin/env python3
"""Display-only adapter. Never publishes hardware states or control commands."""

import math
import time


def convert_target(side, names, positions, gripper_scale=0.03):
    """Convert named retargeting output to URDF joints, including finger units."""
    if side not in ("left", "right"):
        raise ValueError("unknown arm")
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("invalid joint names/positions")
    values = dict(zip(names, positions))
    arm_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    required = arm_names + [f"{side}_gripper_joint"]
    result = [float(values[name]) for name in required]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("non-finite target")
    if not 0 <= result[-1] <= 1:
        raise ValueError("gripper target outside [0, 1]")
    if not math.isfinite(gripper_scale) or gripper_scale <= 0:
        raise ValueError("invalid gripper scale")
    # The other finger is a mimic joint handled by robot_state_publisher.
    return arm_names + [f"openarm_{side}_finger_joint1"], result[:-1] + [result[-1] * gripper_scale]


def fresh_targets(cache, now, timeout=0.5):
    return all(side in cache and 0 <= now - cache[side][0] <= timeout
               for side in ("left", "right"))


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class TargetView(Node):
        def __init__(self):
            super().__init__("exo_target_view_adapter")
            self.declare_parameter("gripper_scale", 0.03)
            self.scale = float(self.get_parameter("gripper_scale").value)
            self.cache = {}
            self.ready = None
            self.publisher = self.create_publisher(JointState, "/exo_target_view/joint_states", 10)
            self.subscriptions_ = [self.create_subscription(
                JointState, f"/{side}_arm/joint_command",
                lambda msg, side=side: self.receive(side, msg), 10)
                for side in ("left", "right")]
            self.timer = self.create_timer(1 / 30, self.publish)

        def receive(self, side, msg):
            try:
                names, positions = convert_target(side, msg.name, msg.position, self.scale)
            except (KeyError, ValueError, TypeError):
                self.cache.pop(side, None)
                return
            self.cache[side] = (time.monotonic(), names, positions)

        def publish(self):
            ready = fresh_targets(self.cache, time.monotonic())
            if ready != self.ready:
                if ready:
                    self.get_logger().info("显示外骨骼映射目标（不是实际反馈）；仅显示，不发送控制。")
                else:
                    self.get_logger().warning("等待双臂有效映射目标或数据已超过0.5秒；停止更新，画面可能保留旧姿态。")
                self.ready = ready
            if not ready:
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            for side in ("left", "right"):
                _, names, positions = self.cache[side]
                msg.name.extend(names)
                msg.position.extend(positions)
            self.publisher.publish(msg)

    rclpy.init()
    node = TargetView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
