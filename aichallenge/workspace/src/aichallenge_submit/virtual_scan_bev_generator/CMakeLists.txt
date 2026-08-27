cmake_minimum_required(VERSION 3.8)
project(virtual_scan_bev_generator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(virtual_scan_bev_rasterizer SHARED
  src/bev_rasterizer.cpp
)

ament_auto_add_executable(virtual_scan_bev_generator_node
  src/virtual_scan_bev_generator_node.cpp
)
target_link_libraries(virtual_scan_bev_generator_node virtual_scan_bev_rasterizer)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_bev_rasterizer test/test_bev_rasterizer.cpp)
  target_link_libraries(test_bev_rasterizer virtual_scan_bev_rasterizer)
endif()

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
