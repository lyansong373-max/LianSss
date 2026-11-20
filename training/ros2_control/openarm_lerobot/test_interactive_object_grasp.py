import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import interactive_object_grasp as grasp


class Terminal:
    def isatty(self):
        return True


class InteractiveObjectGraspTests(unittest.TestCase):
    def test_pi0_keeps_velocity_and_step_limits(self):
        self.assertEqual(len(grasp.TASKS), 6)
        self.assertEqual(grasp.PI0_MAX_JOINT_VELOCITY_RAD_S, 0.25)
        self.assertEqual(grasp.PI0_MAX_STEP_RAD, 0.0125)

    def test_each_object_uses_its_own_act_checkpoint(self):
        self.assertEqual(set(grasp.ACT_POLICIES), set(grasp.TASKS))
        for name in grasp.TASKS:
            with self.subTest(task=name):
                self.assertEqual(grasp.check_act_policy(name), grasp.ACT_POLICIES[name])

    def test_pi0_checkpoint_requires_three_cameras_and_16_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory)
            (policy / "model.safetensors").touch()
            (policy / "policy_preprocessor.json").touch()
            (policy / "policy_postprocessor.json").touch()
            config = {
                "type": "pi0",
                "input_features": {
                    "observation.state": {"shape": [16]},
                    **{f"observation.images.camera{i}": {"shape": [3, 480, 640]}
                       for i in range(1, 4)},
                },
                "output_features": {"action": {"shape": [16]}},
            }
            (policy / "config.json").write_text(json.dumps(config))
            self.assertEqual(grasp.check_policy(policy, "pi0"), policy)
            config["input_features"]["observation.images.camera3"]["shape"] = [3, 224, 224]
            (policy / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "三路相机"):
                grasp.check_policy(policy, "pi0")

    def test_read_only_bridge_state_check(self):
        keys = [f"joint_{i}" for i in range(16)]
        packet = json.dumps({"state": {key: 0.0 for key in keys}}).encode()

        class Receiver:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def bind(self, address):
                self.address = address

            def settimeout(self, seconds):
                self.timeout = seconds

            def recvfrom(self, _size):
                return packet, ("127.0.0.1", 12345)

        receiver = Receiver()
        clock = iter(i * 0.01 for i in range(1000))
        with mock.patch.object(grasp.socket, "socket", return_value=receiver), \
             mock.patch.object(grasp.time, "monotonic", side_effect=lambda: next(clock)):
            result = grasp.read_bridge_start_pose(50110, keys, [0.0] * 16)
        self.assertEqual(result["max_arm_drift_rad"], 0.0)
        self.assertEqual(receiver.address, ("127.0.0.1", 50110))
        self.assertEqual(receiver.timeout, 0.3)

    def test_start_pose_rejects_sag_and_movement(self):
        reference = [0.0] * 16
        ok = grasp.check_start_pose(reference, [reference] * 30)
        self.assertEqual(ok["max_arm_drift_rad"], 0.0)
        sag = list(reference)
        sag[10] = -0.051
        with self.assertRaisesRegex(RuntimeError, "右臂关节3"):
            grasp.check_start_pose(reference, [sag] * 30)
        moving = list(reference)
        moving[10] = 0.03
        with self.assertRaisesRegex(RuntimeError, "仍在移动"):
            grasp.check_start_pose(reference, [reference, moving] * 15)
        with self.assertRaisesRegex(RuntimeError, "足够"):
            grasp.check_start_pose(reference, [reference] * 19)

    def test_every_plan_matches_one_demonstration_start(self):
        for name, (_, plan) in grasp.TASKS.items():
            with self.subTest(task=name):
                reference = grasp.reference_observation(plan)
                self.assertEqual(len(reference["observation"]), 16)
                self.assertEqual(len(reference["first_action"]), 16)
                self.assertGreaterEqual(reference["episode_index"], 0)

    def test_task_aliases(self):
        self.assertEqual(grasp.resolve_task("抓取抽纸"), "抽纸")
        self.assertEqual(grasp.resolve_task("抓取白色胶带"), "双面胶")
        self.assertEqual(grasp.resolve_task("red_tape"), "红色胶带")
        self.assertIsNone(grasp.resolve_task("抓取水瓶"))

    def test_no_execute_does_not_start_any_process(self):
        with mock.patch("sys.argv", ["interactive_object_grasp.py"]), \
             mock.patch.object(grasp.subprocess, "run") as run, \
             mock.patch.object(grasp.subprocess, "Popen") as popen:
            self.assertEqual(grasp.main(), 0)
        run.assert_not_called()
        popen.assert_not_called()

    def test_pi0_one_program_curls_then_rolls_out_with_conservative_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0
            policy = Path(directory) / "pi0_policy"

            def fake_run(command, **_kwargs):
                if "lerobot.scripts.lerobot_rollout" in command:
                    root = next(p.split("=", 1)[1] for p in command
                                if p.startswith("--dataset.root="))
                    video_dir = Path(root) / "videos"
                    video_dir.mkdir(parents=True)
                    for camera in ("camera1", "camera2", "camera3"):
                        (video_dir / f"{camera}.mp4").touch()

            with mock.patch("sys.argv", ["interactive_pi0_object_grasp.py", "--policy-kind", "pi0",
                                         "--execute", "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["透明胶带", "/start", "/quit"]), \
                 mock.patch.object(grasp, "check_pi0_policy", return_value=policy), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run", side_effect=fake_run) as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge) as popen, \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp, "read_bridge_start_pose",
                                   return_value={"actual": [0.0] * 16,
                                                 "max_arm_drift_rad": 0.0}), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            self.assertEqual(run.call_count, 2)
            self.assertIn("curl", run.call_args_list[0].args[0])
            rollout = run.call_args_list[1].args[0]
            self.assertIn(f"--policy.path={policy}", rollout)
            self.assertIn("--strategy.type=episodic", rollout)
            self.assertIn("--dataset.video=true", rollout)
            command = popen.call_args.args[0]
            self.assertIn("safe_mode:=true", command)
            self.assertIn("dry_run:=false", command)
            self.assertIn("max_joint_velocity_rad_s:=0.25", command)
            self.assertIn("max_step_rad:=0.0125", command)
            self.assertIn("enforce_policy_soft_limits:=false", command)
            self.assertFalse(any(part.startswith("max_target_error_rad:=") for part in command))
            self.assertFalse(any(part.startswith("max_tracking_error_rad:=") for part in command))
            self.assertFalse(any(part.startswith("safety_plan_path:=") for part in command))
            self.assertTrue(any(part.startswith("startup_pose_path:=") for part in command))
            self.assertIn("max_joint_state_age_s:=0.3", command)

    def test_pi0_tissue_bridge_disables_soft_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0

            def fake_run(command, **_kwargs):
                if "lerobot.scripts.lerobot_rollout" in command:
                    root = next(part.split("=", 1)[1] for part in command
                                if part.startswith("--dataset.root="))
                    video_dir = Path(root) / "videos"
                    video_dir.mkdir(parents=True)
                    for camera in ("camera1", "camera2", "camera3"):
                        (video_dir / f"{camera}.mp4").touch()

            with mock.patch("sys.argv", ["interactive_pi0_object_grasp.py", "--policy-kind", "pi0",
                                         "--execute", "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["抽纸", "/start", "/quit"]), \
                 mock.patch.object(grasp, "check_pi0_policy", return_value=Path(directory)), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge) as popen, \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp, "read_bridge_start_pose",
                                   return_value={"actual": [0.0] * 16,
                                                 "max_arm_drift_rad": 0.0}), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            command = popen.call_args.args[0]
            self.assertIn("enforce_policy_soft_limits:=false", command)
            self.assertIn("max_joint_velocity_rad_s:=0.25", command)
            self.assertIn("max_step_rad:=0.0125", command)
            self.assertFalse(any(part.startswith("safety_plan_path:=") for part in command))

    def test_task_curls_before_start_and_next_task_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0
            inputs = ["抓取抽纸", "/start", "抓取红色胶带", "/start", "/quit"]
            def fake_run(command, **_kwargs):
                if "lerobot.scripts.lerobot_rollout" in command:
                    root = next(p.split("=", 1)[1] for p in command
                                if p.startswith("--dataset.root="))
                    video_dir = Path(root) / "videos"
                    video_dir.mkdir(parents=True)
                    for camera in ("camera1", "camera2", "camera3"):
                        (video_dir / f"{camera}.mp4").touch()
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=inputs), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run", side_effect=fake_run) as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge) as popen, \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp, "read_bridge_start_pose",
                                   return_value={"actual": [0.0] * 16,
                                                 "max_arm_drift_rad": 0.0}) as start_check, \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            self.assertEqual(run.call_count, 4)
            first, second, third, fourth = [call.args[0] for call in run.call_args_list]
            self.assertIn("curl", first)
            self.assertIn("--from-current", first)
            self.assertIn("--execute", first)
            self.assertIn("--task=Pick up the whole tissue package and place it in the receiving box.", second)
            self.assertIn(f"--policy.path={grasp.ACT_POLICIES['抽纸']}", second)
            self.assertIn("--strategy.type=episodic", second)
            self.assertIn("curl", third)
            self.assertIn("--task=Pick up the red tape and place it in the receiving box.", fourth)
            self.assertIn(f"--policy.path={grasp.ACT_POLICIES['红色胶带']}", fourth)
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(start_check.call_count, 2)
            bridge_command = popen.call_args_list[0].args[0]
            self.assertIn("safe_mode:=true", bridge_command)
            self.assertIn("enforce_policy_soft_limits:=true", bridge_command)
            self.assertIn("max_joint_velocity_rad_s:=0.30", bridge_command)
            self.assertIn("max_step_rad:=0.015", bridge_command)
            self.assertIn("gripper_command_tolerance:=0.25", bridge_command)
            self.assertIn("max_target_error_rad:=0.40", bridge_command)
            self.assertIn("envelope_margin_rad:=0.15", bridge_command)
            self.assertIn("startup_arm_tolerance_rad:=0.05", bridge_command)
            self.assertTrue(any(part.startswith("startup_pose_path:=") for part in bridge_command))
            self.assertTrue(any(part.startswith("safety_plan_path:=") for part in bridge_command))

    def test_record_mode_uses_one_local_episodic_dataset_and_three_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0

            def fake_run(command, **_kwargs):
                if "lerobot.scripts.lerobot_rollout" in command:
                    root = next(p.split("=", 1)[1] for p in command
                                if p.startswith("--dataset.root="))
                    from pathlib import Path
                    video_dir = Path(root) / "videos"
                    video_dir.mkdir(parents=True)
                    for camera in ("camera1", "camera2", "camera3"):
                        (video_dir / f"{camera}.mp4").touch()

            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--record-videos", "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["抓取抽纸", "/start", "/quit"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run", side_effect=fake_run) as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge), \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp, "read_bridge_start_pose",
                                   return_value={"actual": [0.0] * 16,
                                                 "max_arm_drift_rad": 0.0}), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            rollout = run.call_args_list[1].args[0]
            self.assertIn("--strategy.type=episodic", rollout)
            self.assertIn("--dataset.num_episodes=1", rollout)
            self.assertIn("--dataset.push_to_hub=false", rollout)
            self.assertIn("--dataset.video=true", rollout)
            self.assertIn("--dataset.single_task=Pick up the whole tissue package and place it in the receiving box.", rollout)

    def test_quit_at_curl_does_not_grasp(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["抓取抽纸", "/quit"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run") as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge), \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            self.assertEqual(run.call_count, 1)
            self.assertIn("curl", run.call_args.args[0])
            bridge.send_signal.assert_called_once()

    def test_clear_tape_has_bounded_object_specific_envelope_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["透明胶带", "/quit"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run") as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge) as popen, \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            self.assertEqual(run.call_count, 1)
            command = popen.call_args.args[0]
            self.assertIn("envelope_margin_rad:=0.30", command)
            self.assertIn("safe_mode:=true", command)
            self.assertIn("max_target_error_rad:=0.40", command)
            self.assertIn("max_tracking_error_rad:=0.20", command)
            self.assertIn("max_joint_velocity_rad_s:=0.30", command)

    def test_black_tape_uses_shared_target_error_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            bridge.wait.return_value = 0
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["黑色胶带", "/quit"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run") as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge) as popen, \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 0)
            self.assertEqual(run.call_count, 1)
            command = popen.call_args.args[0]
            self.assertIn("max_target_error_rad:=0.40", command)
            self.assertIn("envelope_margin_rad:=0.15", command)
            self.assertIn("max_tracking_error_rad:=0.20", command)
            self.assertIn("max_joint_velocity_rad_s:=0.30", command)
            self.assertIn("max_step_rad:=0.015", command)
            self.assertIn("safe_mode:=true", command)

    def test_sag_at_start_prevents_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = mock.Mock()
            bridge.poll.return_value = None
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["抓取抽纸", "/start"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run") as run, \
                 mock.patch.object(grasp.subprocess, "Popen", return_value=bridge), \
                 mock.patch.object(grasp, "checked_curl_pose", return_value=[0.0] * 16), \
                 mock.patch.object(grasp, "read_bridge_start_pose",
                                   side_effect=RuntimeError("弯举后偏移")), \
                 mock.patch.object(grasp.time, "sleep"):
                self.assertEqual(grasp.main(), 1)
            self.assertEqual(run.call_count, 1)
            bridge.send_signal.assert_called_once()

    def test_failed_curl_never_starts_bridge_or_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["抓取抽纸"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp.subprocess, "run",
                                   side_effect=grasp.subprocess.CalledProcessError(1, "curl")) as run, \
                 mock.patch.object(grasp.subprocess, "Popen") as popen:
                self.assertEqual(grasp.main(), 1)
            self.assertEqual(run.call_count, 1)
            popen.assert_not_called()

    def test_reference_mismatch_prevents_any_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("sys.argv", ["interactive_object_grasp.py", "--execute",
                                         "--output-root", directory]), \
                 mock.patch.object(grasp.sys, "stdin", Terminal()), \
                 mock.patch("builtins.input", side_effect=["抓取抽纸"]), \
                 mock.patch.object(grasp.subprocess, "check_output", return_value="{}"), \
                 mock.patch.object(grasp, "reference_observation",
                                   side_effect=ValueError("no matching first frame")), \
                 mock.patch.object(grasp.subprocess, "run") as run, \
                 mock.patch.object(grasp.subprocess, "Popen") as popen:
                self.assertEqual(grasp.main(), 1)
            run.assert_not_called()
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
