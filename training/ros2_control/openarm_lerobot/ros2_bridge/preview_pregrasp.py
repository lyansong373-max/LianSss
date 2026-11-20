#!/usr/bin/env python3
"""Ask MoveIt to PLAN a single pregrasp; never execute or switch controllers.

Run with ROS2 Humble's /usr/bin/python3 after sourcing the workspace.
Planning success does not certify calibration, environment geometry, or grasp success.
"""

import argparse
import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetMotionPlan, GetPlanningScene
from rosidl_runtime_py.convert import message_to_ordereddict
from shape_msgs.msg import SolidPrimitive


def make_request(target):
    if target.get("purpose") != "pregrasp_plan_only" or target.get("execution_allowed") is not False:
        raise ValueError("只接受 gpt_grasp_prepare.py 生成的预抓取预览目标")
    arm = target["arm"]
    if arm not in ("left", "right") or target["link_name"] != f"openarm_{arm}_hand":
        raise ValueError("机械臂或手部 link 不匹配")
    p, q = target["position_m"], target["orientation_xyzw"]
    if (len(p) != 3 or len(q) != 4 or not all(math.isfinite(x) for x in [*p, *q])
            or not math.isclose(sum(x*x for x in q), 1, abs_tol=1e-4)):
        raise ValueError("无效目标位置或四元数")
    if not isinstance(target["frame_id"], str) or not target["frame_id"].strip():
        raise ValueError("目标缺少坐标系")
    request = GetMotionPlan.Request()
    motion = request.motion_plan_request
    motion.group_name = f"{arm}_arm"
    motion.start_state.is_diff = True
    motion.num_planning_attempts = 3
    motion.allowed_planning_time = 5.0
    motion.max_velocity_scaling_factor = 0.1
    motion.max_acceleration_scaling_factor = 0.1
    constraint = Constraints()
    position = PositionConstraint()
    position.header.frame_id = target["frame_id"]
    position.link_name = target["link_name"]
    position.weight = 1.0
    sphere = SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.003])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, p)
    pose.orientation.w = 1.0
    position.constraint_region.primitives = [sphere]
    position.constraint_region.primitive_poses = [pose]
    orientation = OrientationConstraint()
    orientation.header.frame_id = target["frame_id"]
    orientation.link_name = target["link_name"]
    orientation.orientation.x, orientation.orientation.y, orientation.orientation.z, orientation.orientation.w = map(float, q)
    orientation.absolute_x_axis_tolerance = 0.03
    orientation.absolute_y_axis_tolerance = 0.03
    orientation.absolute_z_axis_tolerance = 0.03
    orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
    orientation.weight = 1.0
    constraint.position_constraints = [position]
    constraint.orientation_constraints = [orientation]
    motion.goal_constraints = [constraint]
    return request


def call(node, service_type, name, request):
    client = node.create_client(service_type, name)
    try:
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError(f"MoveIt 服务不可用：{name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15)
        if not future.done():
            future.cancel()
            raise TimeoutError(f"服务超时：{name}")
        result = future.result()
        if result is None:
            raise RuntimeError(f"服务无返回结果：{name}")
        return result
    finally:
        node.destroy_client(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--board-object-id", required=True,
                        help="规划场景中已建模的洞洞板碰撞物体 ID")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    target = json.loads(args.target.read_text())
    request = make_request(target)
    rclpy.init()
    node = rclpy.create_node("openarm_pregrasp_preview")
    try:
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = scene_request.components.WORLD_OBJECT_GEOMETRY
        scene = call(node, GetPlanningScene, "/get_planning_scene", scene_request).scene
        board = next((obj for obj in scene.world.collision_objects if obj.id == args.board_object_id), None)
        if board is None or not (board.primitives or board.meshes or board.planes):
            raise ValueError("规划场景中没有指定洞洞板的碰撞几何体；请先按实测尺寸建模")
        response = call(node, GetMotionPlan, "/plan_kinematic_path", request).motion_plan_response
        code = response.error_code.val
        if code != response.error_code.SUCCESS or not response.trajectory.joint_trajectory.points:
            raise RuntimeError(f"预抓取规划失败，MoveIt error_code={code}")
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "preview_only_not_for_execution", "execution_allowed": False,
                       "target": target, "board_object_id": args.board_object_id,
                       "motion_plan_response": message_to_ordereddict(response)}, stream,
                      ensure_ascii=False, indent=2, allow_nan=False)
        print(f"预抓取规划已保存到 {args.out}，没有执行运动。规划成功不代表能取下挂钩上的物品。")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
