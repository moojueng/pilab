import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("s_nav_core")

    world_path = os.path.join(pkg_share, "worlds", "grid_world.sdf")
    urdf_path = os.path.join(pkg_share, "urdf", "ssm_bot.urdf")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    gazebo = ExecuteProcess(
        cmd=["gazebo", "--verbose", world_path, "-s", "libgazebo_ros_factory.so"],
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        output="screen",
    )

    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "ssm_bot",
            "-topic", "robot_description",
            "-x", "0",
            "-y", "0",
            "-z", "0.1",
        ],
        output="screen",
    )

    navigation_node = Node(
        package="s_nav_core",
        executable="navigation_node",
        parameters=[
            {
                "map_path": "/home/mj/my_research/ssm_nav_ws/maps/unseen/unseen_map_01.csv",
                "dataset_path": "/home/mj/my_research/ssm_nav_ws/datasets/ssm_nav/unseen.csv",
                "camera_topic": "/camera/image_raw",
                "model_path": "/home/mj/my_research/ssm_nav_ws/models/ssm_policy_action.onnx",
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        navigation_node,
    ])

