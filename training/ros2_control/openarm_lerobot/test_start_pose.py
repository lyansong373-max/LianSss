"""Offline tests: fake robots only, no sockets, cameras or ROS init."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from record_with_start_pose import prepare_record_loop
from start_pose import (
    ARM_KEYS, KEYS, ReturnSettings, interpolated_pose, load_pose,
    motion_duration, move_to_start, save_pose,
)


def pose(joint_value=0.0):
    return {key: joint_value if key in ARM_KEYS else 0.5 for key in KEYS}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRobot:
    name = "openarm_ros2"

    def __init__(self, clock, follow=True):
        self.clock = clock
        self.pose = pose()
        self.follow = follow
        self.commands = []

    def get_observation(self):
        return self.pose.copy()

    def send_action(self, action):
        self.commands.append((self.clock(), action.copy()))
        if self.follow:
            self.pose = action.copy()


class StartPoseTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.robot = FakeRobot(self.clock)
        self.target = pose(0.4)
        self.settings = ReturnSettings(minimum_duration_s=0.2, max_velocity_rad_s=2,
                                       settle_s=0.05, settle_timeout_s=0.3,
                                       handover_timeout_s=0.3)

    def move(self, **kwargs):
        return move_to_start(self.robot, self.target, settings=self.settings,
                             clock=self.clock, sleep=self.clock.sleep, **kwargs)

    def test_smooth_move_finishes_at_taught_pose_and_respects_velocity(self):
        self.move()
        self.assertEqual(self.robot.pose, self.target)
        for (before_t, before), (after_t, after) in zip(self.robot.commands, self.robot.commands[1:]):
            for key in ARM_KEYS:
                self.assertLessEqual(abs(after[key] - before[key]) / (after_t - before_t), 2.000001)
                self.assertGreaterEqual(after[key], 0)
                self.assertLessEqual(after[key], self.target[key])

    def test_long_distance_increases_duration(self):
        self.assertGreaterEqual(motion_duration(pose(), pose(2), self.settings), 1.5)
        self.assertEqual(interpolated_pose(pose(), self.target, 2), self.target)

    def test_unresponsive_hardware_aborts(self):
        self.robot.follow = False
        with self.assertRaises((RuntimeError, TimeoutError)):
            self.move()

    def test_following_error_aborts_before_new_command(self):
        self.robot.pose = pose(3)
        self.robot.follow = False
        with self.assertRaisesRegex(RuntimeError, "跟随误差"):
            self.move()
        self.assertGreater(self.robot.commands[-1][1][ARM_KEYS[0]], 2.6)

    def test_cancel_before_motion_sends_nothing(self):
        with self.assertRaises(KeyboardInterrupt):
            self.move(stopped=lambda: True)
        self.assertEqual(self.robot.commands, [])

    def test_cancel_during_motion_does_not_send_final_target(self):
        with self.assertRaises(KeyboardInterrupt):
            self.move(stopped=lambda: self.clock() >= 0.1)
        self.assertLess(self.robot.commands[-1][1][ARM_KEYS[0]], self.target[ARM_KEYS[0]])

    def test_stale_teleop_prevents_motion(self):
        teleop = Mock()
        teleop.get_action.side_effect = RuntimeError("stale")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.move(teleop=teleop)
        self.assertEqual(self.robot.commands, [])

    def test_handover_waits_for_alignment_and_never_sends_unaligned_exo_target(self):
        teleop = Mock()
        teleop.get_action.return_value = pose(1.5)
        with self.assertRaisesRegex(TimeoutError, "未对齐"):
            self.move(teleop=teleop)
        self.assertTrue(all(command[ARM_KEYS[0]] <= 0.4 for _, command in self.robot.commands))

    def test_aligned_handover_completes(self):
        teleop = Mock()
        teleop.get_action.return_value = self.target.copy()
        self.move(teleop=teleop)
        self.assertEqual(self.robot.pose, self.target)

    def test_invalid_saved_pose_prevents_commands(self):
        self.target[ARM_KEYS[0]] = float("nan")
        with self.assertRaises(ValueError):
            self.move()
        self.assertEqual(self.robot.commands, [])

    def test_pose_file_validates_and_backs_up_previous_teaching(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.json"
            save_pose(path, pose())
            save_pose(path, self.target)
            self.assertEqual(load_pose(path), self.target)
            self.assertEqual(len(list(Path(directory).glob("*.bak"))), 1)
            data = json.loads(path.read_text())
            del data["positions"][ARM_KEYS[0]]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_pose(path)

    def test_record_hook_only_prepares_recorded_episodes_and_rerecords(self):
        original = Mock()

        def loop(robot, events, fps, dataset=None, teleop=None, timer=None):
            original(dataset)

        wrapped = prepare_record_loop(loop, self.target)
        events = {"stop_recording": False, "exit_early": True}
        timer = Mock()
        with patch("record_with_start_pose.move_to_start") as prepare:
            wrapped(self.robot, events, 30, dataset=object(), teleop=Mock(), timer=timer)
            wrapped(self.robot, events, 30, dataset=None, teleop=Mock())
            wrapped(self.robot, events, 30, dataset=object(), teleop=Mock(), timer=timer)
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(original.call_count, 3)
            self.assertEqual(timer.restart.call_count, 2)
            self.assertFalse(events["exit_early"])

    def test_record_hook_supports_opaque_upstream_decorator(self):
        original = Mock()

        def opaque(*args, **kwargs):
            return original(*args, **kwargs)

        wrapped = prepare_record_loop(opaque, self.target)
        with patch("record_with_start_pose.move_to_start") as prepare:
            wrapped(robot=self.robot, events={}, fps=30, dataset=object(), teleop=Mock())
            wrapped(robot=self.robot, events={}, fps=30, dataset=None, teleop=Mock())
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(original.call_count, 2)


if __name__ == "__main__":
    unittest.main()
