import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from openarm_lerobot_bridge import (OpenArmLeRobotBridge, action_envelope,
                                   bounded_arm_target, bounded_gripper_target,
                                   check_startup_pose,
                                   load_startup_pose)


LIMITS = [(-1.5, 1.5)] * 7


def bounded(target, actual=None, previous=None, elapsed=1 / 30):
    return bounded_arm_target(
        target, actual or [0.0] * 7, previous, elapsed, LIMITS,
        max_target_error_rad=0.25, max_tracking_error_rad=0.15,
        max_velocity_rad_s=0.25, max_step_rad=0.0125,
    )


class BridgeSafetyTests(unittest.TestCase):
    def test_gripper_tolerance_clips_only_finite_small_overshoot(self):
        self.assertEqual(bounded_gripper_target(1.25, 0.25), 1.0)
        self.assertEqual(bounded_gripper_target(-0.25, 0.25), 0.0)
        self.assertEqual(bounded_gripper_target(0.4, 0.25), 0.4)
        with self.assertRaisesRegex(ValueError, "原始=1.2510"):
            bounded_gripper_target(1.251, 0.25)
        with self.assertRaisesRegex(ValueError, "原始=nan"):
            bounded_gripper_target(float("nan"), 0.25)
        with self.assertRaisesRegex(ValueError, "原始=None"):
            bounded_gripper_target(None, 0.25)
        with self.assertRaises(ValueError):
            bounded_gripper_target(1.0, 0.51)

    def test_rate_limit_at_30_hz(self):
        result = bounded([0.2] * 7)
        self.assertTrue(all(math.isclose(x, 0.25 / 30) for x in result))
        self.assertLess(max(result), 0.0125)

    def test_delayed_packet_cannot_jump(self):
        result = bounded([0.2] * 7, elapsed=10.0)
        self.assertTrue(all(x <= 0.0125 for x in result))

    def test_reject_distant_model_target(self):
        with self.assertRaisesRegex(ValueError, "关节1 目标=0.2510 rad，实测=0.0000 rad"):
            bounded([0.251] * 7)

    def test_startup_pose_guard(self):
        check_startup_pose([0.0] * 8, [0.0] * 8, "right", 0.05, 0.15)
        sag = [0.0] * 8
        sag[2] = -0.051
        with self.assertRaisesRegex(ValueError, "right关节3"):
            check_startup_pose(sag, [0.0] * 8, "right", 0.05, 0.15)
        with self.assertRaisesRegex(ValueError, "夹爪"):
            check_startup_pose([0.0] * 7 + [0.151], [0.0] * 8,
                               "right", 0.05, 0.15)

    def test_startup_pose_must_be_successful_actual_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"status": "curl_preparation_completed",
                                        "final_actual": [0.1] * 16}))
            self.assertEqual(load_startup_pose(path), [0.1] * 16)
            path.write_text(json.dumps({"status": "stopped",
                                        "final_actual": [0.1] * 16}))
            with self.assertRaises(ValueError):
                load_startup_pose(path)

    def test_reject_tracking_error(self):
        with self.assertRaisesRegex(ValueError, "跟踪误差"):
            bounded([0.2] * 7, previous=[0.151] * 7)

    def test_disabled_soft_limits_still_rate_limit_and_check_urdf(self):
        result = bounded_arm_target(
            [1.2] * 7, [0.0] * 7, [0.5] * 7, 1 / 30, LIMITS,
            max_target_error_rad=None, max_tracking_error_rad=None,
            max_velocity_rad_s=0.25, max_step_rad=0.01,
        )
        self.assertTrue(all(math.isclose(x, 0.5 + 0.25 / 30) for x in result))
        for invalid in ([1.51] * 7, [float("nan")] * 7):
            with self.assertRaises(ValueError):
                bounded_arm_target(
                    invalid, [0.0] * 7, None, 1 / 30, LIMITS,
                    max_target_error_rad=None, max_tracking_error_rad=None,
                    max_velocity_rad_s=0.25, max_step_rad=0.01,
                )

    def test_reject_invalid_and_out_of_limits(self):
        for target in ([float("nan")] * 7, [1.51] * 7):
            with self.assertRaises(ValueError):
                bounded(target)
        with self.assertRaises(ValueError):
            bounded([0.1] * 6)
        with self.assertRaises(ValueError):
            bounded([0.1] * 7, elapsed=0)


class FakeBridge:
    _apply_action = OpenArmLeRobotBridge._apply_action
    _latch_safety = OpenArmLeRobotBridge._latch_safety

    def __init__(self, publishers=1):
        self.lock = threading.Lock()
        self.safe_mode = True
        self.enforce_policy_soft_limits = True
        self.safety_fault = None
        self.limits = {f"openarm_{side}_joint{i}": (-1.5, 1.5)
                       for side in ("left", "right") for i in range(1, 8)}
        self.last_cmd = {"left": None, "right": None}
        self.last_cmd_at = {"left": None, "right": None}
        self.max_target_error = 0.25
        self.max_tracking_error = 0.15
        self.max_joint_velocity = 0.12
        self.max_step = 0.006
        self.arm_envelope = None
        self.startup_pose = None
        self.startup_arm_tolerance = 0.05
        self.startup_gripper_tolerance = 0.15
        self.gripper_command_tolerance = 0.05
        self.dry_run = False
        self.arm_pub = {"left": mock.Mock(), "right": mock.Mock()}
        self._logger = mock.Mock()
        self._publishers = publishers

    def _state_side(self, side):
        return [0.0] * 7, 0.0

    def count_publishers(self, topic):
        return self._publishers

    def get_logger(self):
        return self._logger

    def _send_gripper(self, side, value):
        pass


class BridgeIntegrationTests(unittest.TestCase):
    def action(self, value):
        return {**{f"{side}_joint{i}.pos": value
                   for side in ("left", "right") for i in range(1, 8)},
                "left_gripper.pos": 1.0, "right_gripper.pos": 1.0}

    def test_valid_action_is_rate_limited_before_publish(self):
        bridge = FakeBridge()
        bridge._apply_action(self.action(0.2))
        self.assertIsNone(bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_called_once()
            self.assertTrue(all(abs(x) <= 0.006 for x in
                                publisher.publish.call_args.args[0].data))

    def test_moderate_gripper_overshoot_is_clipped_without_extra_travel(self):
        bridge = FakeBridge()
        bridge.gripper_command_tolerance = 0.25
        action = self.action(0.1)
        action["right_gripper.pos"] = 1.2
        with mock.patch.object(bridge, "_send_gripper") as send_gripper:
            bridge._apply_action(action)
        self.assertIsNone(bridge.safety_fault)
        send_gripper.assert_any_call("right", 1.0)
        bridge.get_logger().warn.assert_called_once()

    def test_large_gripper_overshoot_still_latches(self):
        bridge = FakeBridge()
        bridge.gripper_command_tolerance = 0.25
        action = self.action(0.1)
        action["right_gripper.pos"] = 1.251
        bridge._apply_action(action)
        self.assertIn("原始=1.2510", bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()

    def test_outlier_latches_and_publishes_nothing(self):
        bridge = FakeBridge()
        bridge._apply_action(self.action(0.5))
        self.assertIn("模型目标偏离", bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()

    def test_pi0_soft_limits_off_keeps_hard_guards_and_rate_limit(self):
        bridge = FakeBridge()
        bridge.enforce_policy_soft_limits = False
        bridge.arm_envelope = {side: [(-0.01, 0.01)] * 7 for side in ("left", "right")}
        bridge.last_cmd = {side: [0.5] * 7 for side in ("left", "right")}
        bridge._apply_action(self.action(1.2))
        self.assertIsNone(bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_called_once()
            self.assertTrue(all(0.5 < x <= 0.506 for x in publisher.publish.call_args.args[0].data))
        blocked = FakeBridge(publishers=2)
        blocked.enforce_policy_soft_limits = False
        blocked._apply_action(self.action(1.2))
        self.assertIn("竞争发布者", blocked.safety_fault)

    def test_sag_before_first_action_latches_without_publishing(self):
        bridge = FakeBridge()
        bridge.startup_pose = [0.0] * 16
        def sag_on_right(side):
            return (([0.0, 0.0, -0.06] + [0.0] * 4) if side == "right" else [0.0] * 7), 0.0
        with mock.patch.object(bridge, "_state_side", side_effect=sag_on_right):
            bridge._apply_action(self.action(0.1))
        self.assertIn("[right] 首次动作前实际姿态已偏离", bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()
        bridge._apply_action(self.action(0.1))
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()

    def test_competing_publisher_latches(self):
        bridge = FakeBridge(publishers=2)
        bridge._apply_action(self.action(0.1))
        self.assertIn("竞争发布者", bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()

    def test_demonstration_envelope_blocks_unrelated_pose(self):
        plan = (Path(__file__).resolve().parents[4] /
                "data/demonstrations/tissue_to_blue_box_demonstrations/replay_grasp_down50_left50mm.json")
        bridge = FakeBridge()
        bridge.arm_envelope = action_envelope(plan, 0.10)
        self.assertEqual(len(bridge.arm_envelope["left"]), 7)
        bridge._apply_action(self.action(0.2))
        self.assertIn("示教关节范围", bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()

    def test_envelope_fault_reports_joint_target_bound_and_excess(self):
        bridge = FakeBridge()
        bridge.arm_envelope = {side: [(-0.3, 0.3)] * 7
                               for side in ("left", "right")}
        bridge.arm_envelope["right"][5] = (-0.1, 0.1)
        action = self.action(0.0)
        action["right_joint6.pos"] = 0.15
        bridge._apply_action(action)
        self.assertIn("[right] 模型目标超出该物品的示教关节范围：关节6", bridge.safety_fault)
        self.assertIn("目标=0.1500 rad", bridge.safety_fault)
        self.assertIn("允许=[-0.1000, 0.1000] rad", bridge.safety_fault)
        self.assertIn("超出=0.0500 rad", bridge.safety_fault)
        for publisher in bridge.arm_pub.values():
            publisher.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
