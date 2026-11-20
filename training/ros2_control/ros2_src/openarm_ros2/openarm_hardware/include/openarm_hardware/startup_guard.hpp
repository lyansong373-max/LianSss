#pragma once

#include <linux/can.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace openarm_hardware {

inline can_filter startup_feedback_filter(canid_t id) {
  if (id > CAN_SFF_MASK) throw std::runtime_error("Invalid startup feedback CAN ID");
  // CAN_ERR_FLAG in can_mask routes a SocketCAN subscription to the error
  // receive list, rather than merely excluding error frames. Do NOT add it.
  // Error reception is separately controlled by CAN_RAW_ERR_FILTER (off here).
  return {id, CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG};
}

// Cooperative, process-independent ownership in the CAN network namespace.
// Keep the socket open for the entire hardware object's lifetime. No lock file
// can be unlinked accidentally; process death releases the abstract socket.
// Legacy drivers and other CAN utilities do NOT participate in this lock.
class CanOwnership {
 public:
  CanOwnership() = default;
  CanOwnership(const CanOwnership&) = delete;
  CanOwnership& operator=(const CanOwnership&) = delete;
  ~CanOwnership() { if (fd_ >= 0) ::close(fd_); }

  void acquire(const std::string& interface) {
    if (fd_ >= 0) throw std::runtime_error("CAN ownership already acquired");
    if (interface.empty() || interface.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.:") != std::string::npos)
      throw std::runtime_error("Invalid CAN interface name");
    const std::string name = "openarm.hardware.can." + interface;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (name.size() + 1 > sizeof(address.sun_path))
      throw std::runtime_error("CAN ownership name too long");
    std::memcpy(address.sun_path + 1, name.data(), name.size());
    const int fd = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0) throw std::runtime_error("Cannot create CAN ownership socket");
    if (::bind(fd, reinterpret_cast<sockaddr*>(&address),
               offsetof(sockaddr_un, sun_path) + 1 + name.size()) != 0) {
      const std::string reason = std::strerror(errno);
      ::close(fd);
      throw std::runtime_error("Cannot own " + interface + ": " + reason +
                               "; another OpenArm hardware instance may be running");
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

struct StartupSample {
  double position = 0;
  double velocity = 0;
  double torque = 0;
  double received_at = -1;
  double first_position = 0;
  unsigned count = 0;
};

// Motor-radian checks, including the gripper. Initial zero-filled cache is
// never feedback: only explicitly received and decoded samples count.
class StartupFeedback {
 public:
  explicit StartupFeedback(size_t count) : samples_(count) {
    if (count != 7 && count != 8) throw std::runtime_error("Invalid motor count");
  }
  void accept(size_t index, double q, double dq, double tau, double now) {
    if (index >= samples_.size() || !std::isfinite(q) || !std::isfinite(dq) ||
        !std::isfinite(tau) || !std::isfinite(now) || now < 0)
      throw std::runtime_error("Invalid startup motor feedback");
    if (std::abs(dq) > 0.1)
      throw std::runtime_error("Motor moving during startup; refusing activation");
    auto& sample = samples_[index];
    if (sample.count && (now < sample.received_at || std::abs(q - sample.first_position) > 0.02))
      throw std::runtime_error("Motor moved during startup; refusing activation");
    if (!sample.count) sample.first_position = q;
    sample.position = q;
    sample.velocity = dq;
    sample.torque = tau;
    sample.received_at = now;
    ++sample.count;
  }
  bool ready(double now) const {
    return std::isfinite(now) && std::all_of(samples_.begin(), samples_.end(),
        [now](const StartupSample& s) {
          return s.count >= 3 && now >= s.received_at && now - s.received_at <= 0.1;
        });
  }
  const std::vector<StartupSample>& samples() const { return samples_; }

 private:
  std::vector<StartupSample> samples_;
};

// All arguments are injected so failure paths can be tested without CAN or ROS.
// Snapshot returns already validated joint positions; hold seeds both ROS
// command interfaces and sends the corresponding measured-position command.
inline void activate_measured_hold(
    const std::function<std::vector<double>()>& snapshot,
    const std::function<void(const std::vector<double>&)>& hold,
    const std::function<void()>& enable,
    const std::function<void()>& disable,
    const std::vector<double>& max_change) {
  bool command_attempted = false;
  try {
    const auto before = snapshot();
    if (before.size() != max_change.size() || before.empty())
      throw std::runtime_error("Invalid startup snapshot size");
    for (size_t i = 0; i < before.size(); ++i)
      if (!std::isfinite(before[i]) || !std::isfinite(max_change[i]) || max_change[i] <= 0)
        throw std::runtime_error("Invalid startup snapshot/threshold");
    // Preload a fresh hold target BEFORE enable, rather than restoring a stale
    // motor target. Firmware behavior still needs supervised hardware testing.
    command_attempted = true;
    hold(before);
    enable();
    const auto after = snapshot();
    if (after.size() != before.size()) throw std::runtime_error("Missing post-enable feedback");
    for (size_t i = 0; i < before.size(); ++i)
      if (!std::isfinite(after[i]) || std::abs(after[i] - before[i]) > max_change[i])
        throw std::runtime_error("Pose changed on enable; disabling motors");
    hold(after);
  } catch (...) {
    if (command_attempted) { try { disable(); } catch (...) {} }
    throw;
  }
}

}  // namespace openarm_hardware
