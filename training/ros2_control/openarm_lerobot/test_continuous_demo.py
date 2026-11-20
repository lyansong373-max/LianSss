"""Offline fake-clock tests; no ROS, sockets, cameras, or robot commands."""
import unittest
from unittest.mock import Mock, patch

from record_continuous_demo import (
    DemoActionProcessor, countdown_to_control,
    handle_demo_key, prepare_demo_loop,
)
from start_pose import ARM_KEYS, GRIPPER_KEYS, KEYS, positions


def pose(joint=0.0, grip=0.0):
    return {k: joint if k in ARM_KEYS else grip for k in KEYS}


class Clock:
    def __init__(self, on_sleep=lambda: None):
        self.t = 0.0
        self.on_sleep = on_sleep

    def now(self):
        return self.t

    def sleep(self, duration):
        self.t += duration
        self.on_sleep()


class Robot:
    name = 'openarm_ros2'

    def get_observation(self):
        return pose()

    def send_action(self, action):
        raise AssertionError('Countdown must not send any action')


class Teleop:
    name = 'openarm_exo'

    def get_action(self):
        return pose(joint=2.0)  # Deliberately far from actual pose.


class DemoTests(unittest.TestCase):
    def test_countdown_waits_15s_without_alignment_gate_or_actions(self):
        clock = Clock()
        result = countdown_to_control(Robot(), Teleop(), {}, clock=clock.now, sleep=clock.sleep)
        self.assertGreaterEqual(clock.t, 15)
        self.assertLess(clock.t, 15.1)
        self.assertEqual(result, pose())

    def test_cancel_during_countdown_sends_nothing(self):
        events = {}
        clock = Clock(lambda: events.update(stop_recording=True))
        with self.assertRaises(KeyboardInterrupt):
            countdown_to_control(Robot(), Teleop(), events, clock=clock.now, sleep=clock.sleep)

    def test_stale_exo_or_robot_aborts_countdown(self):
        for cls, method in ((Robot, 'get_observation'), (Teleop, 'get_action')):
            with patch.object(cls, method, side_effect=RuntimeError('stale')):
                with self.assertRaisesRegex(RuntimeError, 'stale'):
                    countdown_to_control(Robot(), Teleop(), {})

    def test_small_negative_feedback_is_preserved_not_commanded(self):
        raw = pose()
        raw['right_gripper.pos'] = -0.02644683416969433
        self.assertEqual(positions(raw, allow_closed_gripper_feedback=True), raw)
        with self.assertRaises(ValueError):
            positions(raw)
        processor = DemoActionProcessor(lambda pair: pair[0], {})
        self.assertEqual(processor((pose(), raw))['right_gripper.pos'], 0.0)
        self.assertLess(raw['right_gripper.pos'], 0)

    def test_feedback_bounds_remain_strict(self):
        for key in GRIPPER_KEYS:
            for value in (-0.030001, 1.000001, float('nan'), float('inf')):
                with self.subTest(key=key, value=value):
                    raw = pose()
                    raw[key] = value
                    with self.assertRaises(ValueError):
                        positions(raw, allow_closed_gripper_feedback=True)

    def test_unbounded_feedback_preserves_raw_values_in_both_phases(self):
        for recording in (False, True):
            processor = DemoActionProcessor(lambda pair: pair[0], {})
            processor.recording = recording
            for value in (-100.0, -0.030001, 1.168736, 1.204533, 1.25, 1.250001, 100.0):
                raw = pose(grip=value)
                self.assertEqual(processor((pose(grip=1), raw)), pose(grip=1))
                self.assertEqual(raw, pose(grip=value))
                self.assertEqual(positions(raw, check_gripper_range=False), raw)
                with self.assertRaises(ValueError):
                    positions(raw)  # Commands and saved poses are not relaxed.

    def test_unbounded_feedback_still_rejects_nonfinite(self):
        for key in KEYS:
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(key=key, value=value):
                    raw = pose()
                    raw[key] = value
                    processor = DemoActionProcessor(lambda pair: pair[0], {})
                    with self.assertRaises(ValueError):
                        processor((pose(), raw))

    def test_unbounded_feedback_still_requires_all_channels(self):
        for key in KEYS:
            raw = pose()
            del raw[key]
            with self.assertRaisesRegex(ValueError, '缺少'):
                DemoActionProcessor(lambda pair: pair[0], {})((pose(), raw))

    def test_countdown_accepts_open_feedback_without_sending_actions(self):
        clock = Clock()
        robot = Robot()
        with patch.object(robot, 'get_observation', return_value=pose(grip=1.5)), \
                patch.object(robot, 'send_action', wraps=robot.send_action) as send:
            actual = countdown_to_control(robot, Teleop(), {}, duration_s=0.1,
                                          clock=clock.now, sleep=clock.sleep)
            send.assert_not_called()
        self.assertEqual(actual, pose(grip=1.5))

    def test_original_targets_pass_through_without_extra_speed_or_range_scaling(self):
        clock = Clock()
        processor = DemoActionProcessor(lambda pair: pair[0], {}, clock=clock.now)
        clock.sleep(0.04)
        target = pose(2, 1)
        first = processor((target, pose()))
        self.assertEqual(first, target)
        self.assertIsNot(first, target)
        processor.recording = True
        clock.sleep(20)
        self.assertEqual(processor((pose(-1, 0), pose())), pose(-1, 0))

    def test_invalid_target_never_relaxed(self):
        processor = DemoActionProcessor(lambda pair: pair[0], {})
        for value in (-0.01, 1.000001, 1.25, float('inf')):
            with self.assertRaises(ValueError):
                processor((pose(grip=value), pose(grip=1.5)))
        with self.assertRaises(ValueError):
            processor((pose(joint=float('nan')), pose()))

    def test_gripper_opening_and_long_closure_never_end_recording(self):
        clock = Clock()
        events = {'demo_phase': 'recording', 'exit_early': False}
        processor = DemoActionProcessor(lambda pair: pair[0], events, clock=clock.now)
        processor.recording = True
        for grip, samples in ((0, 100), (1, 20), (0, 200)):
            for _ in range(samples):
                clock.sleep(0.1)
                self.assertEqual(processor((pose(grip=grip), pose(grip=grip))), pose(grip=grip))
                self.assertFalse(events['exit_early'])
        handle_demo_key('right', events)
        self.assertTrue(events['exit_early'])
        self.assertFalse(events.get('rerecord_episode', False))
        self.assertFalse(events.get('stop_recording', False))

    def test_keyboard_space_ignored_and_backup_controls_phase_scoped(self):
        for phase in ('preparing', 'countdown', 'preroll', 'saving'):
            events = {'demo_phase': phase}
            for key in ('space', 'right', 'n', 'left', 'r'):
                handle_demo_key(key, events)
            self.assertEqual(events, {'demo_phase': phase})
        events['demo_phase'] = 'recording'
        handle_demo_key('space', events)
        self.assertNotIn('exit_early', events)
        handle_demo_key('n', events)
        self.assertTrue(events['exit_early'])
        handle_demo_key('r', events)
        self.assertTrue(events['rerecord_episode'])
        handle_demo_key('q', events)
        self.assertTrue(events['stop_recording'])

    def test_right_arrow_requests_normal_episode_end_in_recording_loop(self):
        events = {}
        def native(**kwargs):
            if kwargs['dataset'] is not None:
                self.assertEqual(events['demo_phase'], 'recording')
                handle_demo_key('right', events)
                self.assertTrue(events['exit_early'])
                self.assertFalse(events.get('rerecord_episode', False))
                self.assertFalse(events.get('stop_recording', False))
                # Native LeRobot consumes this flag and returns to save_episode.
                events['exit_early'] = False
                return 'normal_end'
        with patch('record_continuous_demo.countdown_to_control', return_value=pose()):
            result = prepare_demo_loop(native)(robot=Robot(), teleop=Teleop(), events=events,
                dataset='data', teleop_action_processor=lambda pair: pair[0])
        self.assertEqual(result, 'normal_end')
        self.assertEqual(events['demo_phase'], 'saving')

    def test_hook_runs_unrecorded_five_seconds_before_recording(self):
        calls = []
        def native(**kwargs):
            calls.append(kwargs)
        loop = prepare_demo_loop(native)
        timer = Mock()
        events = {}
        with patch('record_continuous_demo.countdown_to_control', return_value=pose()) as countdown:
            loop(robot=Robot(), teleop=Teleop(), events=events, dataset='dataset',
                 control_time_s=180, timer=timer, teleop_action_processor=lambda pair: pair[0])
            countdown.assert_called_once()
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]['dataset'])
        self.assertEqual(calls[0]['control_time_s'], 5)
        self.assertIsNone(calls[0]['timer'])
        self.assertEqual(calls[1]['dataset'], 'dataset')
        self.assertEqual(calls[1]['control_time_s'], 180)
        self.assertIs(calls[0]['teleop_action_processor'], calls[1]['teleop_action_processor'])
        timer.restart.assert_called_once()

    def test_preroll_cancellation_never_starts_recording(self):
        calls = []
        def native(**kwargs):
            calls.append(kwargs['dataset'])
            kwargs['events']['stop_recording'] = True
        with patch('record_continuous_demo.countdown_to_control', return_value=pose()):
            with self.assertRaises(KeyboardInterrupt):
                prepare_demo_loop(native)(robot=Robot(), teleop=Teleop(), events={}, dataset='data',
                                          teleop_action_processor=lambda pair: pair[0])
        self.assertEqual(calls, [None])

    def test_reset_loop_passthrough(self):
        native = Mock()
        with patch('record_continuous_demo.countdown_to_control') as countdown:
            prepare_demo_loop(native)(robot=Robot(), teleop=Teleop(), events={}, dataset=None)
            countdown.assert_not_called()
        native.assert_called_once()


if __name__ == '__main__':
    unittest.main()
