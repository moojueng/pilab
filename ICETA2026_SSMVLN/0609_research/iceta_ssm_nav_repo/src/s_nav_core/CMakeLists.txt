cmake_minimum_required(VERSION 3.8)
project(s_nav_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(s_nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include"
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
  add_compile_definitions(USE_ONNX_RUNTIME)
else()
  message(STATUS "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}. Using Mock implementation.")
  add_compile_definitions(USE_ONNX_MOCK)
endif()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(grid_policy_eval
  src/grid_policy_eval.cpp
)

add_executable(blind_grid_explore
  src/blind_grid_explore.cpp
)

add_executable(grid_world_sim
  src/grid_world_sim.cpp
)

add_executable(depth_voxel_mapper
  src/depth_voxel_mapper.cpp
)

add_executable(rgbd_frontier_navigator
  src/rgbd_frontier_navigator.cpp
)

add_executable(navigation_node
  src/navigation_node.cpp
  src/GraphManager.cpp
  src/PathPlanner.cpp
  src/SsmInference.cpp
)

ament_target_dependencies(navigation_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  s_nav_msgs
  cv_bridge
)

ament_target_dependencies(depth_voxel_mapper
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  cv_bridge
)

ament_target_dependencies(rgbd_frontier_navigator
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  cv_bridge
)

target_include_directories(navigation_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(navigation_node
  ${OpenCV_LIBS}
)

target_include_directories(depth_voxel_mapper PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(depth_voxel_mapper
  ${OpenCV_LIBS}
)

target_include_directories(rgbd_frontier_navigator PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rgbd_frontier_navigator
  ${OpenCV_LIBS}
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  target_link_libraries(navigation_node
    ${ONNXRUNTIME_LIBRARY}
  )
endif()

install(TARGETS
  grid_policy_eval
  blind_grid_explore
  grid_world_sim
  depth_voxel_mapper
  rgbd_frontier_navigator
  navigation_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  launch
  worlds
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()


target_include_directories(grid_policy_eval PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  target_link_libraries(grid_policy_eval
    ${ONNXRUNTIME_LIBRARY}
  )
endif()
