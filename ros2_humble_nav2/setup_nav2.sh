#!/bin/bash
# setup_nav2.sh - ROS2 Humble 기반 자율주행 환경 패키지 설치 스크립트

echo "[1/3] Updating apt repository..."
sudo apt update

echo "[2/3] Installing ROS2 Navigation2 and related packages..."
sudo apt install -y \
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

echo "[3/3] Exporting TURTLEBOT3_MODEL and sourcing ROS2 setup..."
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

echo "✅ 설치 완료! 새 터미널에서 적용되도록 'source ~/.bashrc' 실행하거나 터미널 재시작하세요."
