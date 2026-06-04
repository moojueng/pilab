import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory("s_nav_core")
    target_goal = LaunchConfiguration("target_goal")
    mission_mode = LaunchConfiguration("mission_mode")
    stop_on_target = LaunchConfiguration("stop_on_target")

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
            "-y", "-3.35",
            "-z", "0.1",
            "-Y", "1.5708",
        ],
        output="screen",
    )

    mapper_params = {
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
        "pixel_stride": 10,
        "max_depth": 2.6,
    }

    depth_voxel_mapper = Node(
        package="s_nav_core",
        executable="depth_voxel_mapper",
        parameters=[mapper_params],
        output="screen",
    )

    rgbd_frontier_navigator = Node(
        package="s_nav_core",
        executable="rgbd_frontier_navigator",
        parameters=[
            {
                "voxel_csv_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/observed_voxels.csv",
                "trajectory_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/trajectory.csv",
                "metrics_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/metrics.csv",
                "graph_nodes_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/runtime_graph_nodes.csv",
                "graph_edges_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/runtime_graph_edges.csv",
                "frontier_features_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/frontier_features.csv",
                "target_events_path": "/home/mj/my_research/ssm_nav_ws/results/gazebo_rgbd/target_events.csv",
                "policy_mode": "heuristic",
                "mission_mode": mission_mode,
                "odom_topic": "/odom",
                "camera_topic": "/camera/image_raw",
                "depth_topic": mapper_params["depth_topic"],
                "cmd_vel_topic": "/cmd_vel",
                "voxel_size": mapper_params["voxel_size"],
                "min_x": mapper_params["min_x"],
                "max_x": mapper_params["max_x"],
                "min_y": mapper_params["min_y"],
                "max_y": mapper_params["max_y"],
                "min_z": mapper_params["min_z"],
                "max_z": mapper_params["max_z"],
                "obstacle_min_z": 0.45,
                "linear_speed": 0.16,
                "angular_speed": 0.55,
                "enable_local_avoidance": False,
                "stop_on_target": stop_on_target,
                "approach_target_candidate": False,
                "enable_target_direction_prior": False,
                "event_log_period_sec": 2.0,
                "obstacle_stop_distance": 0.28,
                "enable_entry_behavior": True,
                "entry_x": 1.5,
                "entry_y": -2.65,
                "target_candidate_ratio_threshold": 0.006,
                "target_confirm_ratio_threshold": 0.055,
                "target_confirm_max_distance": 0.55,
                "target_center_tolerance": 0.18,
                "min_frontier_distance": 0.25,
                "goal_tolerance": 0.12,
                "max_runtime_sec": 240.0,
                "target_goal": target_goal,
            }
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "target_goal",
            default_value="chair",
            description="Semantic target: chair or bed",
        ),
        DeclareLaunchArgument(
            "mission_mode",
            default_value="coverage_patrol",
            description="Mission behavior label: coverage_patrol keeps exploring and logs target sightings",
        ),
        DeclareLaunchArgument(
            "stop_on_target",
            default_value="false",
            description="Stop when the target is confirmed. Default false keeps patrol running and logs events.",
        ),
        gazebo,
        robot_state_publisher,
        spawn_entity,
        TimerAction(period=3.0, actions=[depth_voxel_mapper]),
        TimerAction(period=8.0, actions=[rgbd_frontier_navigator]),
    ])
