"""Offline only: mock ROS, processes and signals; never open CAN or enable."""
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import enable_and_zero as module
from slow_demo_replay import KEYS, NAMES


class FakeChildren:
    def __init__(self, output):
        self.output = output
        self.started = []
        self.code = 0
        self.fail_alive = None

    def start(self, name, command):
        self.started.append((name, command))
        (self.output / (name + '.log')).write_text('test log\n')
        return Mock(poll=lambda: self.code, returncode=self.code)

    def require_alive(self, name):
        if name == self.fail_alive:
            raise RuntimeError('child stopped')


class EnableAndZeroTests(unittest.TestCase):
    def test_udp_profile_is_scoped_and_refuses_custom_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            module.configure_transport()
            self.assertEqual(os.environ['RMW_IMPLEMENTATION'], 'rmw_fastrtps_cpp')
            self.assertEqual(os.environ['FASTRTPS_DEFAULT_PROFILES_FILE'], str(module.DDS_PROFILE))
            self.assertNotIn('ROS_DOMAIN_ID', os.environ)
        for key, value in [('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
                           ('FASTRTPS_DEFAULT_PROFILES_FILE', '/custom.xml'),
                           ('FASTDDS_DEFAULT_PROFILES_FILE', '/custom.xml')]:
            with patch.dict(os.environ, {key: value}, clear=True):
                with self.assertRaises(RuntimeError):
                    module.configure_transport()
                self.assertEqual(os.environ[key], value)

    def test_queries_are_throttled_even_when_responses_immediate(self):
        ready = module.Readiness.__new__(module.Readiness)
        clock = [0.0]
        ready.spin = lambda: clock.__setitem__(0, clock[0] + 0.01)
        ready.require_no_commands = Mock()
        ready.node = Mock()
        ready.node.count_publishers.return_value = 0
        ready.controllers = Mock()
        ready.controllers.service_is_ready.return_value = True
        ready.controllers.call_async.return_value = Mock(
            done=lambda: True, result=lambda: Mock(controller=[]))
        ready.controller_type = Mock()
        ready.state = {}
        with patch.object(module.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, '超时'):
                ready.wait_driver(Mock(), timeout=5)
        self.assertGreaterEqual(ready.controllers.call_async.call_count, 9)
        self.assertLessEqual(ready.controllers.call_async.call_count, 11)

    def test_worker_crash_detected_even_while_launcher_alive(self):
        with tempfile.TemporaryDirectory() as temp:
            children = module.Children(Path(temp))
            children.items['driver'] = (Mock(poll=lambda: None), Mock())
            log = Path(temp) / 'driver.log'
            log.write_text('[INFO] [spawner-3]: process has finished cleanly\n')
            children.require_alive('driver')
            with log.open('a') as stream:
                stream.write('[ERROR] [ros2_control_node-2]: process has ')
            children.require_alive('driver')
            with log.open('a') as stream:
                stream.write('died [pid 123, exit code -6]\n')
            with self.assertRaisesRegex(RuntimeError, '实际工作节点已退出'):
                children.require_alive('driver')
            with self.assertRaises(RuntimeError):
                children.require_alive('driver')

    def test_default_and_help_never_create_ros_or_children(self):
        with patch.object(module, 'Readiness') as readiness, patch.object(module, 'Children') as children:
            self.assertEqual(module.main([]), 0)
            with self.assertRaises(SystemExit) as exit_:
                module.main(['--help'])
            self.assertEqual(exit_.exception.code, 0)
            readiness.assert_not_called()
            children.assert_not_called()

    def test_execute_requires_interactive_terminal_before_any_start(self):
        with patch.object(module.sys.stdin, 'isatty', return_value=False), patch.object(module, 'Children') as children:
            with self.assertRaisesRegex(RuntimeError, '交互终端'):
                module.main(['--execute'])
            children.assert_not_called()

    def test_cancel_confirmation_starts_nothing(self):
        with patch.object(module.sys.stdin, 'isatty', return_value=True), \
             patch.object(module, 'require_files'), patch.object(module, 'require_no_conflicts'), \
             patch.object(module, 'Readiness') as readiness, patch.object(module.socket, 'socket'), \
             patch('builtins.input', return_value='no'), patch.object(module, 'Children') as children:
            self.assertEqual(module.main(['--execute']), 0)
            children.assert_not_called()
            readiness.return_value.close.assert_called_once()

    def test_process_scan_exact_tokens_not_substrings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pid, args in [('900001', b'/usr/bin/python3\0/opt/ros/humble/bin/ros2\0launch\0openarm_bringup\0openarm.bimanual.launch.py\0'),
                              ('900002', b'/usr/bin/python3\0/tmp/slow_demo_replay.py\0zero\0'),
                              ('900003', b'/usr/bin/rg\0ros2_control_node.*test\0'),
                              ('900004', b'/usr/bin/python3\0/tmp/websocket_teleoperator\0')]:
                (root / pid).mkdir()
                (root / pid / 'cmdline').write_bytes(args)
            self.assertEqual({pid for pid, _ in module.conflicting_processes(root)}, {900001, 900002})

    def test_existing_process_refused_not_killed(self):
        with patch.object(module, 'conflicting_processes', return_value=[(123, 'ros2_control_node')]), \
             patch.object(module.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, '123'):
                module.require_no_conflicts()
            kill.assert_not_called()

    def test_feedback_requires_all_finite_recent_values(self):
        state = {name: (0.1, 10) for name in NAMES}
        self.assertTrue(module.fresh_feedback(state, 10.2))
        self.assertFalse(module.fresh_feedback(state, 10.31))
        self.assertFalse(module.fresh_feedback(state, 9.9))
        state.pop(NAMES[-1])
        self.assertFalse(module.fresh_feedback(state, 10.2))
        state[NAMES[-1]] = (math.nan, 10)
        self.assertFalse(module.fresh_feedback(state, 10.2))

    def test_sequence_waits_then_invokes_existing_bounded_zero_worker(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(module, 'verify_zero_result') as verify:
            output = Path(temp)
            children = FakeChildren(output)
            ready = Mock()
            ready.wait_driver.side_effect = lambda _: self.assertEqual([n for n, _ in children.started], ['driver'])
            ready.wait_planner.side_effect = lambda _: self.assertEqual([n for n, _ in children.started], ['driver', 'planner'])
            module.run_sequence(children, ready, output)
            self.assertEqual([n for n, _ in children.started], ['driver', 'planner', 'zero'])
            self.assertIn('allow_trajectory_execution:=false', children.started[1][1])
            self.assertEqual(children.started[2][1][-4:], ['zero', '--out', str(output / 'zero'), '--execute'])
            ready.require_feedback.assert_called_once()
            verify.assert_called_once_with(output / 'zero')

    def test_failed_readiness_never_falls_through_to_homing(self):
        for stage, expected in [('wait_driver', ['driver']), ('wait_planner', ['driver', 'planner']),
                                ('require_no_commands', ['driver', 'planner']),
                                ('require_feedback', ['driver', 'planner'])]:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                children = FakeChildren(Path(temp))
                ready = Mock()
                getattr(ready, stage).side_effect = RuntimeError('not ready')
                with self.assertRaisesRegex(RuntimeError, 'not ready'):
                    module.run_sequence(children, ready, Path(temp))
                self.assertEqual([n for n, _ in children.started], expected)

    def test_failed_zero_or_missing_driver_is_not_success(self):
        for failure in ('zero', 'driver', 'planner'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp, \
                 patch.object(module, 'verify_zero_result') as verify:
                children = FakeChildren(Path(temp))
                if failure == 'zero':
                    children.code = 1
                else:
                    children.fail_alive = failure
                with self.assertRaises(RuntimeError):
                    module.run_sequence(children, Mock(), Path(temp))
                verify.assert_not_called()

    def test_success_requires_saved_zero_and_measured_tolerances(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'result.json').write_text(json.dumps({'status': 'zero_reached_and_saved'}))
            pose = {'encoder_zero_rewritten': False, 'positions': dict.fromkeys(KEYS, 0),
                    'measured_positions': dict.fromkeys(KEYS, 0.001)}
            (path / 'zero_pose.json').write_text(json.dumps(pose))
            module.verify_zero_result(path)
            for field, value in [('encoder_zero_rewritten', True), ('positions', {}),
                                 ('measured_positions', dict.fromkeys(KEYS, 0.5))]:
                bad = {**pose, field: value}
                (path / 'zero_pose.json').write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):
                    module.verify_zero_result(path)

    def test_children_have_separate_sessions_and_reject_duplicate_start(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(module.subprocess, 'Popen') as popen:
            children = module.Children(Path(temp))
            children.start('driver', module.DRIVER)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(popen.call_args.kwargs['stdin'], module.subprocess.DEVNULL)
            with self.assertRaises(RuntimeError):
                children.start('driver', module.DRIVER)
            children.items['driver'][1].close()

    def test_saved_zero_gripper_tolerance_matches_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            (path / 'result.json').write_text(json.dumps({'status': 'zero_reached_and_saved'}))
            for key in ('left_gripper.pos', 'right_gripper.pos'):
                for value,allowed in ((.09323176894164972,True),(.10,True),(-.10,True),
                                      (.100001,False),(-.100001,False)):
                    measured=dict.fromkeys(KEYS,0.)
                    measured[key]=value
                    (path / 'zero_pose.json').write_text(json.dumps({
                        'encoder_zero_rewritten':False,'positions':dict.fromkeys(KEYS,0.),
                        'measured_positions':measured}))
                    if allowed:
                        module.verify_zero_result(path)
                    else:
                        with self.assertRaises(RuntimeError): module.verify_zero_result(path)

    def test_shutdown_signals_only_owned_groups_and_stops_motion_first(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(module.os, 'killpg') as kill:
            children = module.Children(Path(temp))
            for name, pid in [('zero', 910001), ('planner', 910002), ('driver', 910003)]:
                children.items[name] = (Mock(pid=pid, poll=Mock(return_value=0)), Mock())
            children.close()
            self.assertEqual(kill.call_args_list[0].args, (910001, signal.SIGINT))
            self.assertEqual({call.args[0] for call in kill.call_args_list}, {910001, 910002, 910003})
            self.assertNotIn(signal.SIGKILL, [call.args[1] for call in kill.call_args_list])

    def test_successful_keepalive_never_restarts_or_sends_motion(self):
        children = Mock()
        children.items = {}
        children.worker_failure.return_value = None
        children.alive.side_effect = [True, False]
        ready = Mock()
        with self.assertRaisesRegex(RuntimeError, '底层驱动已退出'):
            module.keep_driver(children, ready)
        children.start.assert_not_called()
        ready.spin.assert_called_once()

    def test_crashed_worker_keepalive_warns_without_reenable_or_disable(self):
        children = Mock()
        children.items = {}
        children.alive.side_effect = [True, True, False]
        children.worker_failure.return_value = 'ros2_control_node died'
        with patch('builtins.print') as output:
            with self.assertRaisesRegex(RuntimeError, '底层驱动已退出'):
                module.keep_driver(children, Mock())
        self.assertEqual(sum('硬件控制节点已退出' in str(c) for c in output.call_args_list), 1)
        children.start.assert_not_called()
        children.stop.assert_not_called()
        children.close.assert_not_called()

    def test_failure_keeps_launch_alive_until_operator_interrupts(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(module, 'ROOT', Path(temp)), \
             patch.object(module.sys.stdin, 'isatty', return_value=True), \
             patch.object(module, 'require_files'), patch.object(module, 'require_no_conflicts'), \
             patch.object(module, 'Readiness'), patch.object(module.socket, 'socket'), \
             patch('builtins.input', return_value='ZERO'), patch.object(module, 'Children') as factory, \
             patch.object(module, 'run_sequence', side_effect=RuntimeError('failed preflight')), \
             patch.object(module, 'keep_driver', side_effect=KeyboardInterrupt) as keep:
            factory.return_value.alive.return_value = True
            self.assertEqual(module.main(['--execute']), 130)
            factory.return_value.stop.assert_called_once_with('zero')
            keep.assert_called_once()
            factory.return_value.close.assert_called_once()

    def test_success_keeps_terminal_alive_until_operator_interrupts(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(module, 'ROOT', Path(temp)), \
             patch.object(module.sys.stdin, 'isatty', return_value=True), \
             patch.object(module, 'require_files'), patch.object(module, 'require_no_conflicts'), \
             patch.object(module, 'Readiness'), patch.object(module.socket, 'socket'), \
             patch('builtins.input', return_value='ZERO'), patch.object(module, 'Children') as factory, \
             patch.object(module, 'run_sequence') as sequence, \
             patch.object(module, 'keep_driver', side_effect=KeyboardInterrupt) as keep:
            self.assertEqual(module.main(['--execute']), 130)
            sequence.assert_called_once()
            keep.assert_called_once()
            factory.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
