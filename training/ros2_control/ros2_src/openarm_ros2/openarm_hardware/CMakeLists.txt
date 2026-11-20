# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)
project(openarm_hardware)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(urdf REQUIRED)

# Find openarm_can library
find_package(OpenArmCAN REQUIRED)

add_library(${PROJECT_NAME} SHARED
  src/v10_simple_hardware.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
  include
)

target_link_libraries(${PROJECT_NAME}
  OpenArmCAN::openarm_can
)

ament_target_dependencies(${PROJECT_NAME}
  hardware_interface
  pluginlib
  rclcpp
  rclcpp_lifecycle
  urdf
)

pluginlib_export_plugin_description_file(hardware_interface openarm_hardware.xml)

install(
  TARGETS ${PROJECT_NAME}
  DESTINATION lib
  )

install(DIRECTORY include/
DESTINATION include
)

ament_export_include_directories(
  include
)

ament_export_libraries(
  ${PROJECT_NAME}
)
ament_export_dependencies(
  hardware_interface
  pluginlib
  rclcpp
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_startup_guard test/test_startup_guard.cpp)
  target_include_directories(test_startup_guard PRIVATE include)
  # Built for explicitly requested read-only diagnostics, NEVER registered as a
  # test: the normal ctest suite does not open CAN interfaces.
  add_executable(startup_feedback_probe test/startup_feedback_probe.cpp)
  target_include_directories(startup_feedback_probe PRIVATE include)
  target_link_libraries(startup_feedback_probe ${PROJECT_NAME})
  # This workstation's ROS binaries use a newer system libstdc++ ABI than
  # GCC 11's private link-time library. Match the installed ROS runtime library
  # for this diagnostic executable, without changing global compiler settings.
  find_library(STARTUP_PROBE_STDCXX NAMES libstdc++.so.6
    PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE} NO_DEFAULT_PATH)
  if(STARTUP_PROBE_STDCXX)
    target_link_libraries(startup_feedback_probe ${STARTUP_PROBE_STDCXX})
  endif()
  ament_target_dependencies(startup_feedback_probe hardware_interface rclcpp rclcpp_lifecycle)
endif()

ament_package()
