// Copyright 2025 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "openarm_hardware/v10_simple_hardware.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <sys/socket.h>
#include <urdf/model.h>
#include <openarm/damiao_motor/dm_motor_control.hpp>
#include <thread>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"

namespace openarm_hardware {

OpenArm_v10HW::OpenArm_v10HW() = default;

OpenArm_v10HW::~OpenArm_v10HW() { disable_noexcept(); }

void OpenArm_v10HW::disable_noexcept() noexcept {
  active_ = false;
  if (openarm_ && enable_attempted_) {
    try { openarm_->disable_all(); } catch (...) {}
  }
  enable_attempted_ = false;
}

bool OpenArm_v10HW::parse_config(const hardware_interface::HardwareInfo& info) {
  // Parse CAN interface (default: can0)
  auto it = info.hardware_parameters.find("can_interface");
  can_interface_ = (it != info.hardware_parameters.end()) ? it->second : "can0";

  // Parse arm prefix (default: empty for single arm, "left_" or "right_" for
  // bimanual)
  it = info.hardware_parameters.find("arm_prefix");
  arm_prefix_ = (it != info.hardware_parameters.end()) ? it->second : "";

  // Parse gripper enable (default: true for V10)
  it = info.hardware_parameters.find("hand");
  if (it == info.hardware_parameters.end()) {
    hand_ = true;  // Default to true for V10
  } else {
    // Handle both "true"/"True" and "false"/"False"
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    hand_ = (value == "true");
  }

  // Parse CAN-FD enable (default: true for V10)
  it = info.hardware_parameters.find("can_fd");
  if (it == info.hardware_parameters.end()) {
    can_fd_ = true;  // Default to true for V10
  } else {
    // Handle both "true"/"True" and "false"/"False"
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    can_fd_ = (value == "true");
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Configuration: CAN=%s, arm_prefix=%s, hand=%s, can_fd=%s",
              can_interface_.c_str(), arm_prefix_.c_str(),
              hand_ ? "enabled" : "disabled", can_fd_ ? "enabled" : "disabled");
  return true;
}

void OpenArm_v10HW::generate_joint_names() {
  joint_names_.clear();
  // TODO: read from urdf properly and sort in the future.
  // Currently, the joint names are hardcoded for order consistency to align
  // with hardware. Generate arm joint names: openarm_{arm_prefix}joint{N}
  for (size_t i = 1; i <= ARM_DOF; ++i) {
    std::string joint_name =
        "openarm_" + arm_prefix_ + "joint" + std::to_string(i);
    joint_names_.push_back(joint_name);
  }

  // Generate gripper joint name if enabled
  if (hand_) {
    std::string gripper_joint_name = "openarm_" + arm_prefix_ + "finger_joint1";
    joint_names_.push_back(gripper_joint_name);
    RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Added gripper joint: %s",
                gripper_joint_name.c_str());
  } else {
    RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
                "Gripper joint NOT added because hand_=false");
  }

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Generated %zu joint names for arm prefix '%s'",
              joint_names_.size(), arm_prefix_.c_str());
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_init(
    const hardware_interface::HardwareInfo& info) {
  if (hardware_interface::SystemInterface::on_init(info) !=
      CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
  // Parse configuration
  if (!parse_config(info)) {
    return CallbackReturn::ERROR;
  }

  // Generate joint names based on arm prefix
  generate_joint_names();

  // Read limits from the actual URDF, never from assumed encoder zeros.
  lower_limits_.clear();
  upper_limits_.clear();
  urdf::Model model;
  if (!model.initString(info.original_xml)) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Cannot parse URDF; refusing hardware initialization");
    return CallbackReturn::ERROR;
  }
  for (const auto& name : joint_names_) {
    const auto joint = model.getJoint(name);
    if (!joint || !joint->limits || !std::isfinite(joint->limits->lower) ||
        !std::isfinite(joint->limits->upper) || joint->limits->lower > joint->limits->upper) {
      RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Missing/invalid limits for %s", name.c_str());
      return CallbackReturn::ERROR;
    }
    lower_limits_.push_back(joint->limits->lower);
    upper_limits_.push_back(joint->limits->upper);
  }

  // Acquire BEFORE opening CAN or sending any packets. Held even while inactive.
  try {
    can_ownership_.acquire(can_interface_);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "%s", e.what());
    return CallbackReturn::ERROR;
  }

  // Validate joint count (7 arm joints + optional gripper)
  size_t expected_joints = ARM_DOF + (hand_ ? 1 : 0);
  if (joint_names_.size() != expected_joints) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"),
                 "Generated %zu joint names, expected %zu", joint_names_.size(),
                 expected_joints);
    return CallbackReturn::ERROR;
  }

  // Initialize OpenArm with configurable CAN-FD setting
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Initializing OpenArm on %s with CAN-FD %s...",
              can_interface_.c_str(), can_fd_ ? "enabled" : "disabled");
  openarm_ =
      std::make_unique<openarm::can::socket::OpenArm>(can_interface_, can_fd_);

  // Initialize arm motors with V10 defaults
  openarm_->init_arm_motors(DEFAULT_MOTOR_TYPES, DEFAULT_SEND_CAN_IDS,
                            DEFAULT_RECV_CAN_IDS);

  // Initialize gripper if enabled
  if (hand_) {
    RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Initializing gripper...");
    openarm_->init_gripper_motor(DEFAULT_GRIPPER_MOTOR_TYPE,
                                 DEFAULT_GRIPPER_SEND_CAN_ID,
                                 DEFAULT_GRIPPER_RECV_CAN_ID);
  }

  // Initialize state and command vectors based on generated joint count
  const size_t total_joints = joint_names_.size();
  pos_commands_.resize(total_joints, 0.0);
  vel_commands_.resize(total_joints, 0.0);
  tau_commands_.resize(total_joints, 0.0);
  pos_states_.resize(total_joints, 0.0);
  vel_states_.resize(total_joints, 0.0);
  tau_states_.resize(total_joints, 0.0);

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "OpenArm V10 Simple HW initialized successfully");

  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  // Configuration does not enable motors or send a position target.
  try {
    const auto positions = fresh_startup_positions();
    // Exported ROS handles point into these vectors: do not replace storage.
    std::copy(positions.begin(), positions.end(), pos_states_.begin());
    std::copy(positions.begin(), positions.end(), pos_commands_.begin());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Startup feedback rejected: %s", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
OpenArm_v10HW::export_state_interfaces() {
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_POSITION, &pos_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY, &vel_states_[i]));
    state_interfaces.emplace_back(hardware_interface::StateInterface(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_states_[i]));
  }

  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
OpenArm_v10HW::export_command_interfaces() {
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  // TODO: consider exposing only needed interfaces to avoid undefined behavior.
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_POSITION,
        &pos_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_VELOCITY,
        &vel_commands_[i]));
    command_interfaces.emplace_back(hardware_interface::CommandInterface(
        joint_names_[i], hardware_interface::HW_IF_EFFORT, &tau_commands_[i]));
  }

  return command_interfaces;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "Activating OpenArm V10...");
  active_ = false;
  try {
    std::vector<double> max_change(joint_names_.size(), 0.02);
    if (hand_) max_change.back() = 0.001;
    activate_measured_hold(
        [this]() { return fresh_startup_positions(); },
        [this](const std::vector<double>& q) { seed_hold(q); },
        [this]() { enable_attempted_ = true; openarm_->enable_all(); },
        [this]() { enable_attempted_ = true; disable_noexcept(); },
        max_change);
    active_ = true;
  } catch (const std::exception& e) {
    disable_noexcept();
    RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Activation aborted: %s", e.what());
    return CallbackReturn::ERROR;
  }
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "OpenArm V10 activated: holding measured pose; NO automatic homing or encoder-zero write");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"),
              "Deactivating OpenArm V10...");

  // Disable all motors (like full_arm.cpp exit)
  disable_noexcept();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  openarm_->recv_all();

  RCLCPP_INFO(rclcpp::get_logger("OpenArm_v10HW"), "OpenArm V10 deactivated");
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_shutdown(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  disable_noexcept();
  return CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn OpenArm_v10HW::on_error(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  disable_noexcept();
  // Do not automatically recover/reactivate after a hardware error.
  return CallbackReturn::ERROR;
}

hardware_interface::return_type OpenArm_v10HW::read(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // Receive all motor states
  openarm_->refresh_all();
  openarm_->recv_all();

  // Read arm joint states
  const auto& arm_motors = openarm_->get_arm().get_motors();
  for (size_t i = 0; i < ARM_DOF && i < arm_motors.size(); ++i) {
    pos_states_[i] = arm_motors[i].get_position();
    vel_states_[i] = arm_motors[i].get_velocity();
    tau_states_[i] = arm_motors[i].get_torque();
  }

  // Read gripper state if enabled
  if (hand_ && joint_names_.size() > ARM_DOF) {
    const auto& gripper_motors = openarm_->get_gripper().get_motors();
    if (!gripper_motors.empty()) {
      // TODO the mappings are approximates
      // Convert motor position (radians) to joint value (0-0.044m)
      double motor_pos = gripper_motors[0].get_position();
      pos_states_[ARM_DOF] = motor_radians_to_joint(motor_pos);

      // Unimplemented: Velocity and torque mapping
      vel_states_[ARM_DOF] = 0;  // gripper_motors[0].get_velocity();
      tau_states_[ARM_DOF] = 0;  // gripper_motors[0].get_torque();
    }
  }

  return hardware_interface::return_type::OK;
}

hardware_interface::return_type OpenArm_v10HW::write(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  // ros2_control may call write while inactive: never send default zero targets.
  if (!active_) return hardware_interface::return_type::OK;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    if (!std::isfinite(pos_commands_[i]) || !std::isfinite(vel_commands_[i]) ||
        !std::isfinite(tau_commands_[i])) {
      disable_noexcept();
      RCLCPP_ERROR(rclcpp::get_logger("OpenArm_v10HW"), "Non-finite motor command rejected");
      return hardware_interface::return_type::ERROR;
    }
  }
  send_position_commands();
  openarm_->recv_all(1000);
  return hardware_interface::return_type::OK;
}

void OpenArm_v10HW::send_position_commands() {
  // Control arm motors with MIT control
  std::vector<openarm::damiao_motor::MITParam> arm_params;
  for (size_t i = 0; i < ARM_DOF; ++i) {
    arm_params.push_back({DEFAULT_KP[i], DEFAULT_KD[i], pos_commands_[i],
                          vel_commands_[i], tau_commands_[i]});
  }
  openarm_->get_arm().mit_control_all(arm_params);
  // Control gripper if enabled
  if (hand_ && joint_names_.size() > ARM_DOF) {
    // TODO the true mappings are unimplemented.
    double motor_command = joint_to_motor_radians(pos_commands_[ARM_DOF]);
    openarm_->get_gripper().mit_control_all(
        {{GRIPPER_DEFAULT_KP, GRIPPER_DEFAULT_KD, motor_command, 0, 0}});
  }
}

void OpenArm_v10HW::seed_hold(const std::vector<double>& positions) {
  if (positions.size() != pos_commands_.size()) throw std::runtime_error("Invalid hold size");
  std::copy(positions.begin(), positions.end(), pos_states_.begin());
  std::copy(positions.begin(), positions.end(), pos_commands_.begin());
  std::fill(vel_commands_.begin(), vel_commands_.end(), 0.0);
  std::fill(tau_commands_.begin(), tau_commands_.end(), 0.0);
  send_position_commands();
}

std::vector<double> OpenArm_v10HW::fresh_startup_positions() {
  // A NEW receive-only observer has no cached frames. The CAN library itself
  // exposes zero-filled caches but no per-motor timestamps, so do not use those
  // caches to decide whether activation is safe. No library ABI changes needed.
  openarm::canbus::CANSocket observer(can_interface_, can_fd_);
  auto motors = openarm_->get_arm().get_motors();
  if (hand_) {
    const auto gripper = openarm_->get_gripper().get_motors();
    motors.insert(motors.end(), gripper.begin(), gripper.end());
  }
  if (motors.size() != joint_names_.size()) throw std::runtime_error("Missing motor descriptors");
  std::vector<can_filter> filters;
  for (size_t i = 0; i < motors.size(); ++i) {
    if (motors[i].get_send_can_id() != i + 1 || motors[i].get_recv_can_id() != i + 0x11)
      throw std::runtime_error("Unexpected motor ordering/IDs");
    filters.push_back(startup_feedback_filter(motors[i].get_recv_can_id()));
  }
  if (::setsockopt(observer.get_socket_fd(), SOL_CAN_RAW, CAN_RAW_FILTER,
                   filters.data(), filters.size() * sizeof(can_filter)) != 0)
    throw std::runtime_error("Cannot filter startup feedback");
  openarm_->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
  StartupFeedback feedback(motors.size());
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };
  double next_query = 0;
  while (elapsed() < 1.0) {
    if (elapsed() >= next_query) {
      openarm_->refresh_all();  // state request only, NOT enable or position control
      next_query = elapsed() + 0.02;
    }
    canfd_frame frame{};
    if (observer.is_data_available(1000)) {
      const auto bytes = observer.read_raw_frame(&frame, sizeof(frame));
      if (bytes == CAN_MTU || bytes == CANFD_MTU) {
        if (frame.len == 8 && !(frame.can_id & (CAN_EFF_FLAG | CAN_RTR_FLAG | CAN_ERR_FLAG))) {
          for (size_t i = 0; i < motors.size(); ++i) {
            if (frame.can_id != motors[i].get_recv_can_id()) continue;
            const auto result = openarm::damiao_motor::CanPacketDecoder::parse_motor_state_data(
                motors[i], std::vector<uint8_t>(frame.data, frame.data + frame.len));
            if (result.valid)
              feedback.accept(i, result.position, result.velocity, result.torque, elapsed());
          }
        }
      }
    }
    // Observe at least 100 ms, with >=3 valid fresh replies from EVERY motor.
    if (elapsed() >= 0.1 && feedback.ready(elapsed())) {
      std::vector<double> result;
      for (size_t i = 0; i < motors.size(); ++i) {
        const double motor_q = feedback.samples()[i].position;
        const double q = i == ARM_DOF ? motor_radians_to_joint(motor_q) : motor_q;
        // Allow only small encoder quantisation around the model boundary.
        // Hold the measured position, never snap/clamp it to a zero/limit.
        const double tolerance = i == ARM_DOF ? 0.001 : 0.01;
        if (q < lower_limits_[i] - tolerance || q > upper_limits_[i] + tolerance)
          throw std::runtime_error("Startup position outside URDF bounds: " + joint_names_[i]);
        result.push_back(q);
      }
      return result;
    }
  }
  std::string missing;
  const double now = elapsed();
  for (size_t i = 0; i < motors.size(); ++i) {
    const auto& sample = feedback.samples()[i];
    if (sample.count < 3 || now - sample.received_at > 0.1)
      missing += " " + joint_names_[i];
  }
  throw std::runtime_error("Fresh startup feedback timed out:" + missing);
}

// Gripper mapping helper functions
double OpenArm_v10HW::joint_to_motor_radians(double joint_value) {
  // Joint 0=closed -> motor 0 rad, Joint 0.044=open -> motor -1.0472 rad
  return (joint_value / GRIPPER_JOINT_0_POSITION) *
         GRIPPER_MOTOR_1_RADIANS;  // Scale from 0-0.044 to 0 to -1.0472
}

double OpenArm_v10HW::motor_radians_to_joint(double motor_radians) {
  // Motor 0 rad=closed -> joint 0, Motor -1.0472 rad=open -> joint 0.044
  return GRIPPER_JOINT_0_POSITION *
         (motor_radians /
          GRIPPER_MOTOR_1_RADIANS);  // Scale from 0 to -1.0472 to 0-0.044
}

}  // namespace openarm_hardware

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(openarm_hardware::OpenArm_v10HW,
                       hardware_interface::SystemInterface)
