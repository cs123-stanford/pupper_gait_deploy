cmake_minimum_required(VERSION 3.8)
project(neural_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(neural_controller SHARED src/neural_controller.cpp)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(realtime_tools REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# For the L1+R1 game-mode chord: the controller asks the controller_manager to
# switch itself out for its chord partner. Always present on the robot, since
# controller_manager itself depends on it.
find_package(controller_manager_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(generate_parameter_library REQUIRED)

generate_parameter_library(
  neural_controller_parameters # cmake target name for the parameter library
  src/neural_controller_parameters.yaml # path to input yaml file
)

set(RTNEURAL_EIGEN ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(modules/RTNeural)

target_link_libraries(neural_controller neural_controller_parameters RTNeural)

target_include_directories(neural_controller PRIVATE include)

ament_target_dependencies(
  neural_controller
  controller_interface
  hardware_interface
  pluginlib
  realtime_tools
  rclcpp
  rclcpp_lifecycle
  geometry_msgs
  sensor_msgs
  controller_manager_msgs
  tf2
)

pluginlib_export_plugin_description_file(controller_interface neural_controller.xml)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Cross-language parity check for the gait reference the controller computes
  # for motion-reference policies. Standalone (no gtest) so the same source
  # also runs on a laptop via test/run_host_test.sh in the deploy repo.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gait_reference.cpp)
    add_executable(test_gait_reference test/test_gait_reference.cpp)
    # Pinned explicitly: RTNeural's CMake sets CMAKE_CXX_STANDARD to 14 when it is
    # otherwise unset, and this test (and nlohmann/json) want 17.
    set_target_properties(test_gait_reference PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(test_gait_reference PRIVATE
      include
      modules/RTNeural/modules/json)
    target_compile_definitions(test_gait_reference PRIVATE
      GAIT_GOLDEN_JSON="${CMAKE_CURRENT_SOURCE_DIR}/test/gait_golden.json")
    # Second argument: also check the policy that is actually deployed here. The
    # test skips it if none has been installed yet.
    add_test(NAME gait_reference_parity COMMAND test_gait_reference
      ${CMAKE_CURRENT_SOURCE_DIR}/test/gait_golden.json
      ${CMAKE_CURRENT_SOURCE_DIR}/launch/default_policy.json)
    # Same binary, second fixture: gait cases plus the slot composite cases.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/mixed_jump_golden.json)
      add_test(NAME mixed_jump_reference_parity COMMAND test_gait_reference
        ${CMAKE_CURRENT_SOURCE_DIR}/test/mixed_jump_golden.json
        ${CMAKE_CURRENT_SOURCE_DIR}/launch/default_policy.json)
    endif()
  endif()

  # obs_spec layout resolution, command clip, heading hold.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_command_filter.cpp)
    add_executable(test_command_filter test/test_command_filter.cpp)
    set_target_properties(test_command_filter PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_include_directories(test_command_filter PRIVATE
      include
      modules/RTNeural/modules/json)
    add_test(NAME command_filter COMMAND test_command_filter)
  endif()
endif()

ament_package()
