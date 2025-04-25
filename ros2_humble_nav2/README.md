
![image](https://github.com/user-attachments/assets/08f4a51b-f135-44ff-8f7a-b194d4579b68)

🛍 ROS2 Humble 기반 자유주회 로벌 설정 & 시행 정보 (TurtleBot3 + Nav2 + Cartographer)

가상: Ubuntu 22.04 + ROS2 Humble + Gazebo Classic + RViz + TurtleBot3 + Navigation2 + Cartographer

✅ 설치 전 검사 (TurtleBot3, nav2, slam 패키지)

sudo apt update && sudo apt install -y \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-turtlebot3* \
  ros-humble-cartographer \
  ros-humble-cartographer-ros \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-tf2-tools \
  ros-humble-xacro \
  ros-humble-nav2-map-server

필요 하나라도 빈집 없이 설치 가능한 모든 패키지 포함

Ὄ1 경로 설정 (Bashrc)

# ~/.bashrc 마지막에 추가
export TURTLEBOT3_MODEL=burger
source /opt/ros/humble/setup.bash

🚀 1. Gazebo 프리시스 모델 버그없이 배치

ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

🗺️ 2. SLAM (Cartographer)

ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=true

ros2 run rviz2 rviz2 -d /opt/ros/humble/share/turtlebot3_cartographer/rviz/tb3_cartographer.rviz

2D Pose Estimate 클릭 → 2D Goal Pose 로 자유주회 테스트 가능

📃 3. 명확한 map.pgm / map.yaml 저장

ros2 run nav2_map_server map_saver_cli -f ~/map/map

결과: ~/map/map.yaml, ~/map/map.pgm

🛍 4. Navigation2 (모델 이동) 시작

ros2 launch turtlebot3_navigation2 navigation2.launch.py \
  map:=/home/mjay/map/map.yaml use_sim_time:=true

🔍 5. RViz 자유주회 UI 방지

ros2 run rviz2 rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz

다음 버튼 표시 가능:

Pause | Cancel | Clear Costmap | Start | Waypoint Mode

가장 첫번째가 2D Pose Estimate

Nav2 Goal 검색 클릭 후 이동

⚠️ 문제 해결 핀

문제해결

RViz: map transform error

map.yaml 경로 잘못이나 Cartographer 정지해야 함

Localization inactive

2D Pose Estimate 검색 클릭

Queue is full / base_scan

LaserScan 문제 또는 TF 확인

Navigation 진행 안됨

AMCL 명령 또는 map frame 복잡

🔎 TF / Pose 검색

ros2 topic echo /amcl_pose
ros2 run tf2_tools view_frames

🚀 결과

[스크린캐스트 04-25-2025 10:40:07 AM.webm](https://github.com/user-attachments/assets/5d2754d4-15a5-4159-91f0-c743714d1577)


🌐 참고

https://github.com/ROBOTIS-GIT/turtlebot3

https://navigation.ros.org

https://github.com/ros-planning/navigation2

