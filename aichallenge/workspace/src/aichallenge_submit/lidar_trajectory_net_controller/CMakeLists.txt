cmake_minimum_required(VERSION 3.8)
project(lidar_trajectory_net_controller)

find_package(ament_cmake_auto REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_auto_find_build_dependencies()

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  lidar_trajectory_net_controller/lidar_trajectory_net_controller_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
  ckpt
)
