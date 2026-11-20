#!/usr/bin/env python3
"""
外骨骼-仿真桥接节点
接收 retargeting 后的关节数据 (/left_arm/joint_command, /right_arm/joint_command)
转换为控制器话题 (Float64MultiArray) 和 夹爪动作 (GripperCommand)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from control_msgs.action import GripperCommand
import numpy as np

class ExoskeletonBridgeNode(Node):
    def __init__(self):
        super().__init__('exoskeleton_bridge_node')
        
        # 参数
        self.declare_parameter('gripper_threshold', 0.005) # 夹爪变化的最小阈值
        self.gripper_threshold = self.get_parameter('gripper_threshold').value
        
        self.declare_parameter('gripper_scaling_factor', 1.0) # 夹爪数值缩放因子
        self.gripper_scale = self.get_parameter('gripper_scaling_factor').value

        # ---------------- 左臂 ----------------
        # 订阅
        self.left_arm_sub = self.create_subscription(
            JointState,
            '/left_arm/joint_command',
            self.left_arm_callback,
            10
        )
        
        # 发布: 手臂位置控制
        self.left_arm_pub = self.create_publisher(
            Float64MultiArray,
            '/left_forward_position_controller/commands',
            10
        )
        
        # Action Client: 夹爪控制
        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_gripper_controller/gripper_cmd')
        
        # ---------------- 右臂 ----------------
        # 订阅
        self.right_arm_sub = self.create_subscription(
            JointState,
            '/right_arm/joint_command',
            self.right_arm_callback,
            10
        )
        
        # 发布: 手臂位置控制
        self.right_arm_pub = self.create_publisher(
            Float64MultiArray,
            '/right_forward_position_controller/commands',
            10
        )
        
        # Action Client: 夹爪控制
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        # ---------------- 状态记录 ----------------
        self.last_left_gripper_pos = 0.0  # 初始化为0，避免启动时突然跳变
        self.last_right_gripper_pos = 0.0 # 初始化为0
        self.left_gripper_goal_handle = None
        self.right_gripper_goal_handle = None
        
        # 日志计数器
        self.log_counter_left = 0
        self.log_counter_right = 0
        self.log_interval = 50  # 每50次打印一次日志
        
        self.get_logger().info('Exoskeleton Bridge Node 已启动')

    def left_arm_callback(self, msg: JointState):
        self._process_arm_command(
            msg, 
            self.left_arm_pub, 
            self.left_gripper_client, 
            'left',
            '/left_gripper_controller/gripper_cmd'
        )

    def right_arm_callback(self, msg: JointState):
        self._process_arm_command(
            msg, 
            self.right_arm_pub, 
            self.right_gripper_client, 
            'right',
            '/right_gripper_controller/gripper_cmd'
        )

    def _process_arm_command(self, msg: JointState, arm_pub, gripper_client, side, action_name):
        try:
            # 1. 处理手臂关节 (前7个)
            if len(msg.position) >= 7:
                arm_positions = list(msg.position[:7])
                
                # 发布 Float64MultiArray
                cmds = Float64MultiArray()
                cmds.data = arm_positions
                arm_pub.publish(cmds)
                
                # ---------------- 日志打印 ----------------
                if side == 'left':
                    self.log_counter_left += 1
                    current_counter = self.log_counter_left
                else:
                    self.log_counter_right += 1
                    current_counter = self.log_counter_right
                
                if current_counter % self.log_interval == 0:
                    # 简略打印关节数据 (保留2位小数)
                    short_pos = [f"{x:.2f}" for x in arm_positions[:4]] 
                    self.get_logger().info(
                        f'[{side.upper()}] 收到: {short_pos}... -> 发布: {short_pos}... (Float64MultiArray)'
                    )

            # 2. 处理夹爪
            # 逻辑参考 openarm_hardware_control_node.py
            gripper_position = 0.0
            has_gripper_data = False

            # 方法1：检查是否有第8个位置（夹爪归一化位置0.0-1.0）
            if len(msg.position) >= 8:
                gripper_position = msg.position[7]
                has_gripper_data = True
            
            # 方法2：通过joint_name查找 gripper_joint
            elif f'{side}_gripper_joint' in msg.name:
                try:
                    gripper_idx = msg.name.index(f'{side}_gripper_joint')
                    if gripper_idx < len(msg.position):
                        gripper_position = msg.position[gripper_idx]
                        has_gripper_data = True
                except ValueError:
                    pass

            if has_gripper_data:
                # 这里的 gripper_position 可能是归一化的(0-1)，或者是实际物理值
                # 用户示例中 command: {position: 0.02} 看起来像是物理值 (m)
                # openarm 代码中: finger_position = gripper_position * 0.67 (转换为0-0.044m)
                # 假设接收到的是归一化值(0-1)，需要转换。或者接收到的已经是目标值?
                # 既然 openarm 节点做了转换，说明 retargeting 发出来的是归一化值。
                # 这里我们假设目标控制器接受的是物理值（如0.04m）。
                # 我们使用 openarm 的比例作为参考，或者直接透传如果已经是物理值。
                # 根据用户提供的 action 示例 position: 0.02，这通常是物理单位（米）。
                # 如果 retargeting 发送 0-1，我们需要缩放。
                # 假设 1.0 -> 最大开口。通常 gripper action position 是指开合宽度或者单纯的 joint position。
                # 这里简单假设需要一个缩放因子，或者透传。
                # 考虑到通用性，暂时假设 msg.position 里的值可以直接用于 action，或者做一个简单的映射。
                # openarm 中 1.0 对应 0.044 * 2 (双指) ? 不，是指单指行程。
                
                # 我们沿用 openarm 节点的逻辑：输入 0-1 -> 输出物理值
                # 假设最大行程为 0.04 (4cm)
                target_pos = gripper_position * 0.04 
                
                # 但如果 retargeting 已经发的是物理值，就不应该缩放。
                # 观察 openarm_hardware_control_node.py:
                # finger_position = gripper_position * 0.67  <-- 这看起来是个很大的系数?
                # 实际上 openarm 代码有点奇怪，finger_position 赋值给了 msg.position[7/8]。
                # 如果是模拟环境，通常直接用。
                
                # 为了稳妥，我们这里直接使用 msg.position[7] 的值，但加上一个保护。
                # 如果数值 > 1.0，可能是错误的，或者单位问题。
                # 既然用户给的示例是 0.02，这很小。
                # 假设 retargeting 发过来的是 0-1。
                # 我们这里先按直接透传处理，或者根据 openarm 逻辑调整。
                # 修正：openarm 代码写的是 finger_position = gripper_position * 0.67
                # 稍等，0.67 * 1.0 = 0.67m = 67cm? 这太大了。
                # 也许 openarm 的单位是弧度? 或者那个系数是针对特定舵机的。
                
                # 应用缩放因子
                final_gripper_pos = gripper_position * self.gripper_scale
                
                self._send_gripper_action(gripper_client, final_gripper_pos, side)

        except Exception as e:
            self.get_logger().error(f'Error processing {side} arm command: {e}')

    def _send_gripper_action(self, client, position, side):
        # 检查 Action Server 是否连接
        if not client.server_is_ready():
            # 避免刷屏，可以用 throttle
            # self.get_logger().warn(f'{side} gripper action server not ready', throttle_duration_sec=5)
            return

        # 检查变化阈值
        last_pos = self.last_left_gripper_pos if side == 'left' else self.last_right_gripper_pos
        
        # 如果变化量超过阈值，才发送新命令
        if abs(position - last_pos) > self.gripper_threshold:
            # 发送 Action
            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = float(position)
            goal_msg.command.max_effort = 10.0 # 固定力矩
            
            # 异步发送，不等待结果，以免阻塞
            future = client.send_goal_async(goal_msg)
            # future.add_done_callback(self.goal_response_callback)
            
            # 更新最后位置
            if side == 'left':
                self.last_left_gripper_pos = position
            else:
                self.last_right_gripper_pos = position
                
            self.get_logger().info(
                f'[{side.upper()}] 夹爪动作请求: {position:.4f} (Action: {client._action_name})'
            )

def main(args=None):
    rclpy.init(args=args)
    node = ExoskeletonBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

