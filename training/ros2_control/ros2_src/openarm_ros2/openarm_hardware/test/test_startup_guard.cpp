#include <gtest/gtest.h>
#include <sys/wait.h>

#include <limits>
#include "openarm_hardware/startup_guard.hpp"

using namespace openarm_hardware;

TEST(FeedbackFilter, SubscribesToNormalStandardDataFramesNotErrorList) {
  for (canid_t id = 0x11; id <= 0x18; ++id) {
    const auto filter = startup_feedback_filter(id);
    EXPECT_EQ(filter.can_id, id);
    EXPECT_EQ(filter.can_mask, CAN_SFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG);
    EXPECT_EQ(filter.can_mask & CAN_ERR_FLAG, 0u);
    EXPECT_EQ(id & filter.can_mask, filter.can_id);
    EXPECT_NE((id | CAN_EFF_FLAG) & filter.can_mask, filter.can_id);
    EXPECT_NE((id | CAN_RTR_FLAG) & filter.can_mask, filter.can_id);
  }
  EXPECT_THROW(startup_feedback_filter(0x800), std::runtime_error);
}

TEST(Ownership, DuplicateIsRejectedAndReleasedOnDestruction) {
  const auto name = "test" + std::to_string(getpid());
  {
    CanOwnership owner;
    owner.acquire(name);
    CanOwnership duplicate;
    EXPECT_THROW(duplicate.acquire(name), std::runtime_error);
    CanOwnership other;
    EXPECT_NO_THROW(other.acquire(name + "other"));
    EXPECT_THROW(owner.acquire(name), std::runtime_error);
  }
  CanOwnership replacement;
  EXPECT_NO_THROW(replacement.acquire(name));
}

TEST(Ownership, AnotherProcessCannotAcquire) {
  const auto name = "process" + std::to_string(getpid());
  CanOwnership owner;
  owner.acquire(name);
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try { CanOwnership duplicate; duplicate.acquire(name); _exit(1); }
    catch (const std::runtime_error&) { _exit(0); }
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

TEST(Ownership, ChildExitReleasesOwnership) {
  const auto name = "release" + std::to_string(getpid());
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try { CanOwnership owner; owner.acquire(name); _exit(0); }
    catch (...) { _exit(1); }
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  CanOwnership owner;
  EXPECT_NO_THROW(owner.acquire(name));
}

TEST(Ownership, BadNamesRejected) {
  CanOwnership owner;
  EXPECT_THROW(owner.acquire(""), std::runtime_error);
  EXPECT_THROW(owner.acquire("../can0"), std::runtime_error);
  EXPECT_THROW(owner.acquire(std::string(150, 'x')), std::runtime_error);
}

TEST(Feedback, DefaultZerosAreNotActualFeedback) {
  StartupFeedback feedback(8);
  EXPECT_FALSE(feedback.ready(0));
  EXPECT_FALSE(feedback.ready(0.1));
}

TEST(Feedback, EveryMotorNeedsThreeFreshRepliesIncludingGripper) {
  StartupFeedback feedback(8);
  for (size_t i = 0; i < 7; ++i)
    for (int n = 0; n < 3; ++n) feedback.accept(i, 0.7, 0, 0, 0.01 * n);
  EXPECT_FALSE(feedback.ready(0.03));
  feedback.accept(7, -0.4, 0, 0, 0.01);
  feedback.accept(7, -0.4, 0, 0, 0.02);
  EXPECT_FALSE(feedback.ready(0.03));
  feedback.accept(7, -0.4, 0, 0, 0.03);
  EXPECT_TRUE(feedback.ready(0.04));
  EXPECT_FALSE(feedback.ready(0.2));
  EXPECT_FALSE(feedback.ready(-1));
  EXPECT_FALSE(feedback.ready(std::numeric_limits<double>::quiet_NaN()));
}

TEST(Feedback, RejectNonFiniteMotionInvalidIndexAndBackwardsTime) {
  StartupFeedback feedback(7);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(feedback.accept(0, nan, 0, 0, 0), std::runtime_error);
  EXPECT_THROW(feedback.accept(0, 0, nan, 0, 0), std::runtime_error);
  EXPECT_THROW(feedback.accept(0, 0, 0, nan, 0), std::runtime_error);
  EXPECT_THROW(feedback.accept(0, 0, 0, 0, nan), std::runtime_error);
  EXPECT_THROW(feedback.accept(7, 0, 0, 0, 0), std::runtime_error);
  EXPECT_THROW(feedback.accept(0, 0, 0.11, 0, 0), std::runtime_error);
  feedback.accept(0, 0, 0, 0, 0.1);
  EXPECT_THROW(feedback.accept(0, 0.021, 0, 0, 0.2), std::runtime_error);
  EXPECT_THROW(feedback.accept(0, 0, 0, 0, 0.09), std::runtime_error);
  EXPECT_THROW(StartupFeedback(0), std::runtime_error);
}

TEST(Activation, SeedsMeasuredArmAndOpenGripperBeforeEnableAndAfterFeedback) {
  const std::vector<double> pose{0.1, -0.2, 0.3, 1.2, 0.2, 0.1, -0.1, 0.03};
  std::vector<std::string> events;
  activate_measured_hold(
      [&]() { events.push_back("feedback"); return pose; },
      [&](const auto& q) { events.push_back("hold"); EXPECT_EQ(q, pose); },
      [&]() { events.push_back("enable"); },
      [&]() { events.push_back("disable"); }, std::vector<double>(8, 0.02));
  EXPECT_EQ(events, (std::vector<std::string>{"feedback", "hold", "enable", "feedback", "hold"}));
}

TEST(Activation, MissingPreEnableFeedbackSendsNothing) {
  int commands = 0;
  EXPECT_THROW(activate_measured_hold(
      []() -> std::vector<double> { throw std::runtime_error("timeout"); },
      [&](const auto&) { ++commands; }, [&]() { ++commands; },
      [&]() { ++commands; }, std::vector<double>(8, 0.02)), std::runtime_error);
  EXPECT_EQ(commands, 0);
}

TEST(Activation, MissingPostEnableFeedbackDisables) {
  int snapshots = 0, holds = 0, enables = 0, disables = 0;
  EXPECT_THROW(activate_measured_hold(
      [&]() { if (++snapshots > 1) throw std::runtime_error("timeout"); return std::vector<double>(8, 0.1); },
      [&](const auto&) { ++holds; }, [&]() { ++enables; }, [&]() { ++disables; },
      std::vector<double>(8, 0.02)), std::runtime_error);
  EXPECT_EQ(holds, 1);
  EXPECT_EQ(enables, 1);
  EXPECT_EQ(disables, 1);
}

TEST(Activation, PoseChangeOnEnableDisablesWithoutSendingChangedPose) {
  int snapshots = 0, holds = 0, disables = 0;
  EXPECT_THROW(activate_measured_hold(
      [&]() { return std::vector<double>(8, ++snapshots == 1 ? 0.1 : 0.2); },
      [&](const auto&) { ++holds; }, []() {}, [&]() { ++disables; },
      std::vector<double>(8, 0.02)), std::runtime_error);
  EXPECT_EQ(holds, 1);
  EXPECT_EQ(disables, 1);
}

TEST(Activation, InvalidSnapshotNeverEnables) {
  for (const auto& pose : {std::vector<double>{},
                         std::vector<double>(8, std::numeric_limits<double>::quiet_NaN())}) {
    int commands = 0;
    EXPECT_THROW(activate_measured_hold([&]() { return pose; },
        [&](const auto&) { ++commands; }, [&]() { ++commands; }, [&]() { ++commands; },
        std::vector<double>(8, 0.02)), std::runtime_error);
    EXPECT_EQ(commands, 0);
  }
}

TEST(Activation, HoldOrEnableExceptionAttemptsDisable) {
  for (bool fail_hold : {false, true}) {
    int disables = 0;
    EXPECT_THROW(activate_measured_hold([]() { return std::vector<double>(8, 0.1); },
        [&](const auto&) { if (fail_hold) throw std::runtime_error("send failed"); },
        [&]() { throw std::runtime_error("enable failed"); }, [&]() { ++disables; },
        std::vector<double>(8, 0.02)), std::runtime_error);
    EXPECT_EQ(disables, 1);
  }
}

TEST(Activation, SmallMeasuredChangeUpdatesHoldInsteadOfReturningToZero) {
  int snapshots = 0;
  std::vector<double> final_hold;
  activate_measured_hold([&]() { return std::vector<double>(8, ++snapshots == 1 ? 0.5 : 0.501); },
      [&](const auto& q) { final_hold = q; }, []() {}, []() {}, std::vector<double>(8, 0.02));
  EXPECT_EQ(final_hold, std::vector<double>(8, 0.501));
}

TEST(Activation, GripperHasSeparateChangeThreshold) {
  int snapshots = 0, disables = 0;
  std::vector<double> thresholds(8, 0.02);
  thresholds.back() = 0.001;
  EXPECT_THROW(activate_measured_hold([&]() {
      std::vector<double> q(8, 0);
      q.back() = ++snapshots == 1 ? 0.03 : 0.032;
      return q;
    }, [](const auto&) {}, []() {}, [&]() { ++disables; }, thresholds), std::runtime_error);
  EXPECT_EQ(disables, 1);
}

TEST(Activation, BadPostEnableSnapshotDisables) {
  for (const auto& invalid : {std::vector<double>{},
                            std::vector<double>(8, std::numeric_limits<double>::infinity())}) {
    int snapshots = 0, disables = 0;
    EXPECT_THROW(activate_measured_hold([&]() {
        return ++snapshots == 1 ? std::vector<double>(8, 0) : invalid;
      }, [](const auto&) {}, []() {}, [&]() { ++disables; },
      std::vector<double>(8, 0.02)), std::runtime_error);
    EXPECT_EQ(disables, 1);
  }
}
