"""Offline checks: no cameras, API requests, ROS nodes, or physical movement."""

import copy
import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cv2

from gpt_grasp_prepare import (calibrate_board, codex_environment, locate, locate_codex,
                               project_grasp, require_codex_chatgpt, rotation_xyzw,
                               validate_selection, write_new)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        # Synthetic camera at origin, board z=1; board faces the camera (-z).
        self.calibration = {
            "measured_and_verified": True,
            "camera_matrix": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            "distortion": [0]*5, "T_base_camera": np.eye(4).tolist(),
            "board_origin_base_m": [0, 0, 1], "board_outward_normal_base": [0, 0, -1],
            "image_size": [640, 480], "workspace_bounds_base_m": [[-2, -2, -2], [2, 2, 2]],
        }
        self.profile = {
            "measured_and_verified": True, "grasp_plane_offset_m": 0.1,
            "pregrasp_distance_m": 0.15, "tcp_in_hand_m": [0, 0, 0.1],
            "hand_orientation_xyzw": [0, 0, 0, 1],
        }

    def test_object_plane_corrects_depth_and_parallax(self):
        tcp, pregrasp = project_grasp(self.calibration, self.profile, [420, 240])
        # Intersecting the bare board then subtracting z gives x=.2 (wrong).
        np.testing.assert_allclose(tcp, [0.18, 0, 0.9])
        np.testing.assert_allclose(pregrasp, [0.18, 0, 0.65])

    def test_base_transform_and_rotated_tcp(self):
        self.calibration["T_base_camera"][0][3] = 0.4
        self.profile["tcp_in_hand_m"] = [0.1, 0, 0]
        self.profile["hand_orientation_xyzw"] = [0, 0, np.sqrt(0.5), np.sqrt(0.5)]
        tcp, pregrasp = project_grasp(self.calibration, self.profile, [320, 240])
        np.testing.assert_allclose(tcp, [0.4, 0, 0.9])
        np.testing.assert_allclose(pregrasp, [0.4, -0.1, 0.75])

    def test_refuse_unmeasured_templates(self):
        self.calibration["measured_and_verified"] = False
        with self.assertRaises(ValueError):
            project_grasp(self.calibration, self.profile, [320, 240])

    def test_refuse_invalid_geometry(self):
        for key, value in (
            ("board_outward_normal_base", [0, 0, 1]),
            ("T_base_camera", (np.eye(4)*2).tolist()),
            ("distortion", [float("nan")]*5),
            ("workspace_bounds_base_m", [[0, 0, 0], [0.01, 0.01, 0.01]]),
        ):
            with self.subTest(key=key):
                calibration = copy.deepcopy(self.calibration)
                calibration[key] = value
                with self.assertRaises(ValueError):
                    project_grasp(calibration, self.profile, [420, 240])

    def test_refuse_pixel_outside_image(self):
        for pixel in ([640, 240], [-1, 0], [0, float("nan")]):
            with self.assertRaises(ValueError):
                project_grasp(self.calibration, self.profile, pixel)

    def test_refuse_nonunit_quaternion(self):
        with self.assertRaises(ValueError):
            rotation_xyzw([0, 0, 0, 0])

    def test_parallel_ray_and_plane_behind_camera(self):
        self.calibration["board_outward_normal_base"] = [-1, 0, 0]
        self.calibration["board_origin_base_m"] = [1, 0, 0]
        with self.assertRaises(ValueError):
            project_grasp(self.calibration, self.profile, [320, 240])
        self.calibration["board_outward_normal_base"] = [0, 0, 1]
        self.calibration["board_origin_base_m"] = [0, 0, -1]
        with self.assertRaises(ValueError):
            project_grasp(self.calibration, self.profile, [320, 240])


class SelectionTests(unittest.TestCase):
    def test_accept_unique_target_and_abstention(self):
        validate_selection(dict(status="found", object_id="white_tape", bbox_xyxy=[.1,.2,.4,.8], reason="白胶带"))
        validate_selection(dict(status="ambiguous", object_id="unknown", bbox_xyxy=None, reason="多个目标"))

    def test_refuse_invalid_or_extra_motion_fields(self):
        good = dict(status="found", object_id="white_tape", bbox_xyxy=[.1,.2,.4,.8], reason="白胶带")
        for update in ({"joints": [0]*7}, {"bbox_xyxy": [1, 0, 0, 1]},
                       {"bbox_xyxy": [0, 0, 1, float("nan")]}, {"object_id": "unknown"},
                       {"status": "ambiguous"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_selection({**good, **update})

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"result.json"
            write_new(path, {"result": 1})
            with self.assertRaises(FileExistsError):
                write_new(path, {"result": 2})

    def test_missing_key_does_not_send_images(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(out=Path(directory)/"new", backend="api")
            with patch.dict("os.environ", {}, clear=True), patch("urllib.request.build_opener") as opener:
                with self.assertRaises(ValueError):
                    locate(args)
                opener.assert_not_called()


class CodexBackendTests(unittest.TestCase):
    def test_environment_removes_api_overrides_without_modifying_parent(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy", "CODEX_API_KEY": "dummy",
                                    "OPENAI_BASE_URL": "https://example.invalid", "PATH": "/bin"}, clear=True):
            environment = codex_environment()
            self.assertEqual(environment, {"PATH": "/bin"})
            self.assertEqual(os.environ["OPENAI_API_KEY"], "dummy")

    def test_reject_api_login_without_printing_secret(self):
        result = subprocess.CompletedProcess([], 0, "", "Logged in using an API key - secret_fixture")
        with patch("subprocess.run", return_value=result), self.assertRaises(RuntimeError) as error:
            require_codex_chatgpt("codex", {})
        self.assertNotIn("secret_fixture", str(error.exception))

    def test_cli_request_uses_auth_guard_schema_and_images(self):
        selection = dict(status="found", object_id="white_tape", bbox_xyxy=[.1,.2,.4,.8], reason="候选")
        images = {f"camera{i}": b"image_fixture" for i in range(1,4)}
        args = argparse.Namespace(snapshot=Path("fixture"), instruction="抓取白色胶带", model="gpt-6-astra")
        def fake_run(command, **kwargs):
            self.assertIn("--ignore-user-config", command)
            self.assertIn('forced_login_method="chatgpt"', command)
            self.assertIn("read-only", command)
            self.assertEqual(command.count("--image"), 3)
            self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
            self.assertNotIn("CODEX_API_KEY", kwargs["env"])
            self.assertEqual(kwargs["timeout"], 180)
            self.assertFalse(kwargs.get("shell", False))
            write_new(command[command.index("--output-last-message")+1], selection)
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch("shutil.which", return_value="codex"), patch("gpt_grasp_prepare.load_snapshot", return_value=({}, images)), \
             patch("gpt_grasp_prepare.require_codex_chatgpt") as auth, \
             patch("subprocess.run", side_effect=fake_run), patch("gpt_grasp_prepare.save_selection") as save:
            locate_codex(args)
            auth.assert_called_once()
            self.assertEqual(save.call_args.args[3], selection)
            self.assertEqual(save.call_args.args[4]["auth_mode"], "chatgpt")

    def test_codex_failure_never_falls_back_to_api(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(out=Path(directory)/"new", backend="codex")
            with patch("gpt_grasp_prepare.locate_codex", side_effect=RuntimeError("offline")), \
                 patch("gpt_grasp_prepare.locate_api") as api, self.assertRaises(RuntimeError):
                locate(args)
            api.assert_not_called()
            self.assertFalse(args.out.exists())


class BoardFitTests(unittest.TestCase):
    def test_known_pose_recovered_without_marking_verified(self):
        K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
        # A synthetic plate at base z=1, camera translated .1 m along base x.
        xyz = np.array([[-.2,-.2,1], [.2,-.2,1], [-.2,.2,1], [.2,.2,1],
                        [0,-.2,1], [0,.2,1], [-.1,0,1], [.1,0,1]], dtype=float)
        pixels, _ = cv2.projectPoints(xyz, np.zeros(3), np.array([-.1,0,0]), K, np.zeros(5))
        pairs = [{"pixel_uv": uv.tolist(), "base_xyz_m": point.tolist()}
                 for point, uv in zip(xyz, pixels.reshape(-1,2))]
        manifest = {"cameras": {"camera1": {"sha256": "fixture", "device": "test", "image_size": [640,480]}}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = argparse.Namespace(intrinsics=directory/"intrinsics.json", points=directory/"points.json",
                                      snapshot=directory/"snapshot", out=directory/"calibration.json")
            write_new(args.intrinsics, {"device": "test", "image_size": [640,480],
                                       "camera_matrix": K.tolist(), "distortion": [0]*5})
            write_new(args.points, {"base_frame": "test_base", "source_image_sha256": "fixture",
                                   "fit_points": pairs[:6], "check_points": pairs[6:]})
            with patch("gpt_grasp_prepare.load_snapshot", return_value=(manifest, {})):
                calibrate_board(args)
            result = json.loads(args.out.read_text())
            np.testing.assert_allclose(np.array(result["T_base_camera"])[:3,3], [.1,0,0], atol=1e-6)
            np.testing.assert_allclose(result["board_outward_normal_base"], [0,0,-1])
            self.assertFalse(result["measured_and_verified"])
            self.assertIsNone(result["workspace_bounds_base_m"])
            self.assertLess(max(result["holdout_errors_px"]), 1e-5)

    def test_board_fit_refuses_wrong_image(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            args = argparse.Namespace(intrinsics=directory/"i.json", points=directory/"p.json",
                                      snapshot=directory, out=directory/"out.json")
            write_new(args.intrinsics, {"device": "test", "image_size": [640,480]})
            write_new(args.points, {"source_image_sha256": "old_image"})
            manifest = {"cameras": {"camera1": {"device": "test", "image_size": [640,480], "sha256": "new_image"}}}
            with patch("gpt_grasp_prepare.load_snapshot", return_value=(manifest, {})), self.assertRaises(ValueError):
                calibrate_board(args)
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
