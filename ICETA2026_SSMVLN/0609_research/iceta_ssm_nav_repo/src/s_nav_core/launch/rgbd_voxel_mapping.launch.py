import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("s_nav_core")

    world_path = os.path.join(pkg_share, "worlds", "small_house_world.sdf")
    urdf_path = os.path.join(pkg_share, "urdf", "ssm_bot.urdf")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    gazebo_executable = "gazebo" if os.environ.get("DISPLAY") else "gzserver"
    gazebo = ExecuteProcess(
        cmd=[gazebo_executable, "--verbose", world_path, "-s", "libgazebo_ros_factory.so"],
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
            "-x", "1.5",
            "-y", "-4.05",
            "-z", "0.1",
            "-Y", "1.5708",
        ],
        output="screen",
    )

    depth_voxel_mapper = Node(
        package="s_nav_core",
        executable="depth_voxel_mapper",
        parameters=[
            {
                "depth_topic": "/camera/depth/image_raw",
                "camera_info_topic": "/camera/depth/camera_info",
                "odom_topic": "/odom",
                "output_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/observed_voxels.csv",
                "voxel_size": 0.20,
                "min_x": -4.0,
                "max_x": 7.0,
                "min_y": -4.5,
                "max_y": 4.5,
                "min_z": 0.0,
                "max_z": 2.4,
                "pixel_stride": 8,
                "max_depth": 6.0,
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        depth_voxel_mapper,
    ])
