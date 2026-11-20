"""Isolated, display-only TF tree for exoskeleton retargeting output."""
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    description = xacro.process_file(
        str(Path(get_package_share_directory("openarm_description")) / "urdf/robot/v10.urdf.xacro"),
        mappings={"bimanual": "true", "ros2_control": "false"},
    ).toxml()
    # BOTH TF topics are private: target transforms never enter the real TF tree.
    private_tf = [("/tf", "/exo_target_view/tf"),
                  ("/tf_static", "/exo_target_view/tf_static")]
    return LaunchDescription([
        ExecuteProcess(cmd=["/usr/bin/python3", str(root / "ros2_bridge/exo_target_view.py")],
                       output="screen"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             name="exo_target_state_publisher", output="screen",
             parameters=[{"robot_description": description}],
             remappings=private_tf + [
                 ("joint_states", "/exo_target_view/joint_states"),
                 ("robot_description", "/exo_target_view/robot_description")]),
        Node(package="rviz2", executable="rviz2", name="openarm_exo_target_view",
             arguments=["-d", str(root / "rviz/exo_target.rviz")],
             remappings=private_tf, output="screen"),
    ])
