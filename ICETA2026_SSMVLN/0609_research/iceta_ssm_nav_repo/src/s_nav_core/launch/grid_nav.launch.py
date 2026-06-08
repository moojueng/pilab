from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    gazebo_ros_share = get_package_share_directory("gazebo_ros")
    world_path = "/home/mj/my_research/ssm_nav_ws/src/s_nav_core/worlds/generated_grid_world.sdf"
    turtlebot_model = "/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world": world_path,
            "verbose": "false",
        }.items(),
    )

    spawn_turtlebot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "turtlebot3_waffle_pi",
            "-file", turtlebot_model,
            "-x", "-4.5",
            "-y", "-3.6",
            "-z", "0.05",
        ],
        output="screen",
    )

    return LaunchDescription([
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        gazebo,
        spawn_turtlebot,
    ])

