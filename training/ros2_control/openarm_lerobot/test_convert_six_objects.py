"""Offline tests; no hardware or external downloads."""
import unittest
import json
import os
import tempfile
from pathlib import Path
import numpy as np

from convert_six_objects_to_lerobot import (
    CAMERAS, convert, increasing, nearest_indices, preceding_indices, plan_alignment, safe_path,
)


class AlignmentTests(unittest.TestCase):
    def test_nearest_bounds_and_ties(self):
        np.testing.assert_array_equal(nearest_indices(np.array([1., 2., 3.]),
                                                      np.array([.9, 1.5, 2.8, 3.1])), [0, 0, 2, 2])

    def test_feedback_is_causal(self):
        idx, age = preceding_indices(np.array([1., 1.02, 1.04]), np.array([1.01, 1.04]), .03)
        np.testing.assert_array_equal(idx, [0, 2])
        np.testing.assert_allclose(age, [.01, 0])

    def test_missing_or_stale_feedback_rejected(self):
        for grid in (np.array([.9]), np.array([1.2])):
            with self.assertRaises(ValueError):
                preceding_indices(np.array([1., 1.02]), grid, .08)

    def test_invalid_timestamps_rejected(self):
        for values in ([1, 1], [2, 1], [1, float('nan')], [1]):
            with self.assertRaises(ValueError):
                increasing(values, 'test')

    def test_camera_alignment(self):
        cameras = {f'camera{i}': 10 + np.arange(600) / 30 + i * .004 for i in range(3)}
        grid, images, indices, report = plan_alignment(cameras, np.arange(9.9, 31, .02))
        self.assertEqual(len(grid), 600)
        self.assertAlmostEqual(grid[0], 10.008)
        self.assertTrue(np.all(np.diff(indices) >= 0))
        self.assertLess(report['max_log_age_s'], .021)
        for values in images.values():
            self.assertTrue(np.all(np.diff(values) >= 0))

    def test_frame_gap_rejected(self):
        times = np.arange(600) / 30
        times[300:] += .2
        with self.assertRaises(ValueError):
            plan_alignment({'camera': times}, np.arange(-.1, 21, .02))

    def test_outside_archive_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(Path('/tmp/example_archive'), '../outside')


@unittest.skipUnless(os.environ.get('OPENARM_CONVERTER_INTEGRATION') == '1', 'Opt-in codec integration test')
class EncodingTests(unittest.TestCase):
    def test_two_synthetic_episodes(self):
        import cv2
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory(prefix='openarm_conversion_test_') as temp:
            root = Path(temp)
            source = root / 'synthetic'; source.mkdir()
            (source / '总清单.json').write_text('{}')
            for channel, camera in enumerate(CAMERAS):
                writer = cv2.VideoWriter(str(source / f'{camera}.mp4'),
                                         cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
                self.assertTrue(writer.isOpened())
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame[:, :, channel] = 200
                for _ in range(600):
                    writer.write(frame)
                writer.release()
            samples = []
            for episode, task in enumerate(('抓取螺丝刀', '抓取红色胶带')):
                samples.append({'entry': {'task': task, 'directory': f'ep{episode}'},
                                'path': source,
                                'image_indices': {c: np.arange(600) for c in CAMERAS},
                                'actual': np.full((600, 16), episode + .1, dtype=np.float32),
                                'command': np.full((600, 16), episode + .2, dtype=np.float32),
                                'alignment': {}})
            output = root / 'dataset'
            convert(source, output, 'local/converter_test', samples, [], 'en')
            data = LeRobotDataset('local/converter_test', root=output, video_backend='pyav')
            self.assertEqual(data.num_frames, 1200)
            self.assertEqual(data.num_episodes, 2)
            self.assertFalse(output.with_name('dataset.partial').exists())
            self.assertEqual(json.loads((output / 'conversion_report.json').read_text())['episodes'], 2)
            for channel in range(3):
                image = data[0][f'observation.images.camera{channel+1}'].numpy()
                # OpenCV BGR -> RGB channel reversal must survive H264 encoding.
                self.assertGreater(image[2-channel].mean(), .65)
                self.assertLess(image[(3-channel) % 3].mean(), .05)
            with self.assertRaises(FileExistsError):
                convert(source, output, 'local/converter_test', samples, [], 'en')


if __name__ == '__main__':
    unittest.main()
