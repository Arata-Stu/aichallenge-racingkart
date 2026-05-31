cmake_minimum_required(VERSION 3.8)
project(laserscan_generator)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(laserscan_generator_node
  src/laserscan_generator_node.cpp
)

ament_auto_add_executable(occupancy_map_scan_generator_node
  src/occupancy_map_scan_generator_node.cpp
)
target_include_directories(occupancy_map_scan_generator_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(occupancy_map_scan_generator_node
  ${OpenCV_LIBS}
)

ament_auto_package(INSTALL_TO_SHARE 
  launch 
  map 
  config
)
