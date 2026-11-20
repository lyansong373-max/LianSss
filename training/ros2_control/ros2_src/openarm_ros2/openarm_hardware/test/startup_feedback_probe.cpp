// Explicit read-state-only diagnostic, NOT a ctest and NOT a hardware launcher.
// Reads a full URDF from stdin; invokes init/configure only. Never activates,
// writes command interfaces, enables motors, or creates a controller manager.
#include <hardware_interface/component_parser.hpp>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#include "openarm_hardware/v10_simple_hardware.hpp"

int main() {
  try {
    const std::string xml{std::istreambuf_iterator<char>(std::cin),
                          std::istreambuf_iterator<char>()};
    const auto components = hardware_interface::parse_control_resources_from_urdf(xml);
    if (components.size() != 2) throw std::runtime_error("Expected exactly two arm components");
    std::vector<std::unique_ptr<openarm_hardware::OpenArm_v10HW>> arms;
    for (const auto& component : components) {
      if (component.hardware_class_type != "openarm_hardware/OpenArm_v10HW")
        throw std::runtime_error("Unexpected hardware class");
      auto arm = std::make_unique<openarm_hardware::OpenArm_v10HW>();
      if (arm->on_init(component) != hardware_interface::CallbackReturn::SUCCESS)
        throw std::runtime_error("Read-only initialization failed: " + component.name);
      arms.push_back(std::move(arm));
    }
    for (auto& arm : arms) {
      if (arm->on_configure(rclcpp_lifecycle::State{}) != hardware_interface::CallbackReturn::SUCCESS)
        throw std::runtime_error("Read-only configuration/feedback check failed");
      for (const auto& state : arm->export_state_interfaces())
        if (state.get_interface_name() == "position")
          std::cout << state.get_name() << " = " << state.get_value() << '\n';
    }
    std::cout << "PASS: both arms configured with fresh feedback; NO activation or motion commands sent.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Read-only probe failed: " << e.what() << '\n';
    return 1;
  }
}
