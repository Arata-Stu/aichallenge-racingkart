cmake_minimum_required(VERSION 3.8)
project(rsu_fusion_net_pytorch)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  rsu_fusion_net_pytorch/ros_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_geometry test/test_geometry.py)
  ament_add_pytest_test(test_policy test/test_policy.py)
endif()

ament_package()
