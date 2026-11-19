cmake_minimum_required(VERSION 3.8)
project(gm_chain)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(filters REQUIRED)
find_package(grid_map_filters REQUIRED)
find_package(grid_map_cv REQUIRED)
find_package(pluginlib REQUIRED)
find_package(class_loader REQUIRED)

add_executable(gm_chain src/gm_chain.cpp)
target_compile_features(gm_chain PUBLIC cxx_std_17)
ament_target_dependencies(
  gm_chain
  rclcpp
  grid_map_core
  filters
  grid_map_filters
  grid_map_cv
  pluginlib
  class_loader
)
install(TARGETS gm_chain DESTINATION lib/${PROJECT_NAME})
install(FILES chain.yaml DESTINATION share/${PROJECT_NAME})
ament_package()
