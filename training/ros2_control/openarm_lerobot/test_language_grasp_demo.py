"""Offline CLI routing tests: replace recorder with an argument-capturing stub."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DemoRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Path(__file__).resolve().parent
        stub = self.root / 'python'
        stub.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'if sys.argv[1] == "-":\n'
            f'    os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n'
            'print("CAPTURE=" + json.dumps(sys.argv[1:]))\n'
        )
        stub.chmod(0o700)
        self.env = {**os.environ, 'LEROBOT_BIN': str(self.root),
                    'DEMO_DATASET_ROOT': str(self.root / 'tissue'),
                    'DEMO_DATASET_ID': 'local/tissue',
                    'SCREWDRIVER_DATASET_ROOT': str(self.root / 'screwdriver'),
                    'SCREWDRIVER_DATASET_ID': 'local/screwdriver',
                    'WHITE_TAPE_DATASET_ROOT': str(self.root / 'white_tape'),
                    'WHITE_TAPE_DATASET_ID': 'local/white_tape',
                    'CLEAR_TAPE_DATASET_ROOT': str(self.root / 'clear_tape'),
                    'CLEAR_TAPE_DATASET_ID': 'local/clear_tape',
                    'RED_TAPE_DATASET_ROOT': str(self.root / 'red_tape'),
                    'RED_TAPE_DATASET_ID': 'local/red_tape',
                    'BLACK_TAPE_DATASET_ROOT': str(self.root / 'black_tape'),
                    'BLACK_TAPE_DATASET_ID': 'local/black_tape'}

    def run_mode(self, mode, seconds='180'):
        return subprocess.run(['bash', str(self.project / 'language_grasp.sh'), mode, seconds],
                              env=self.env, capture_output=True, text=True)

    def args_from(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(next(line.removeprefix('CAPTURE=') for line in result.stdout.splitlines()
                               if line.startswith('CAPTURE=')))
        self.assertEqual(Path(args[0]).name, 'record_continuous_demo.py')
        return dict(arg.split('=', 1) for arg in args[1:])

    def test_screwdriver_is_separate_and_records_one_episode(self):
        args = self.args_from(self.run_mode('demo-screwdriver'))
        self.assertEqual(args['--dataset.root'], self.env['SCREWDRIVER_DATASET_ROOT'])
        self.assertEqual(args['--dataset.repo_id'], 'local/screwdriver')
        self.assertEqual(args['--dataset.single_task'], 'Pick up the screwdriver.')
        self.assertEqual(args['--resume'], 'false')
        self.assertEqual(args['--dataset.num_episodes'], '1')
        self.assertEqual(args['--dataset.episode_time_s'], '180')
        self.assertEqual(len(json.loads(args['--robot.cameras'])), 3)
        self.assertFalse(Path(args['--dataset.root']).exists())

    def test_tissue_route_unchanged(self):
        args = self.args_from(self.run_mode('demo-tissue'))
        self.assertEqual(args['--dataset.root'], self.env['DEMO_DATASET_ROOT'])
        self.assertEqual(args['--dataset.single_task'],
                         'Pick up the whole tissue package from the shelf and place it in the blue box.')

    def test_red_tape_separate_and_resume(self):
        args = self.args_from(self.run_mode('demo-red-tape', '120'))
        self.assertEqual(args['--dataset.root'], self.env['RED_TAPE_DATASET_ROOT'])
        self.assertEqual(args['--dataset.repo_id'], 'local/red_tape')
        self.assertEqual(args['--dataset.single_task'],
                         'Pick up the red tape and place it in the blue box.')
        self.assertEqual(args['--dataset.num_episodes'], '1')
        self.assertEqual(args['--dataset.episode_time_s'], '120')
        self.assertEqual(args['--resume'], 'false')
        self.assertEqual(len(json.loads(args['--robot.cameras'])), 3)
        root = self.root / 'red_tape'
        root.mkdir()
        self.assertNotEqual(self.run_mode('demo-red-tape').returncode, 0)
        (root / 'meta').mkdir()
        (root / 'meta/info.json').write_text('{}')
        self.assertEqual(self.args_from(self.run_mode('demo-red-tape'))['--resume'], 'true')
        self.assertNotEqual(self.run_mode('demo-red-tape', '0').returncode, 0)

    def test_black_tape_separate_and_resume(self):
        args = self.args_from(self.run_mode('demo-black-tape', '120'))
        self.assertEqual(args['--dataset.root'], self.env['BLACK_TAPE_DATASET_ROOT'])
        self.assertEqual(args['--dataset.repo_id'], 'local/black_tape')
        self.assertEqual(args['--dataset.single_task'],
                         'Pick up the black tape and place it in the blue box.')
        self.assertEqual(args['--dataset.num_episodes'], '1')
        self.assertEqual(args['--dataset.episode_time_s'], '120')
        self.assertEqual(args['--resume'], 'false')
        self.assertEqual(len(json.loads(args['--robot.cameras'])), 3)
        root = self.root / 'black_tape'
        self.assertFalse(root.exists())
        root.mkdir()
        self.assertNotEqual(self.run_mode('demo-black-tape').returncode, 0)
        (root / 'meta').mkdir()
        (root / 'meta/info.json').write_text('{}')
        self.assertEqual(self.args_from(self.run_mode('demo-black-tape'))['--resume'], 'true')
        self.assertNotEqual(self.run_mode('demo-black-tape', '0').returncode, 0)

    def test_double_sided_tape_name_and_legacy_route(self):
        args = self.args_from(self.run_mode('demo-white-tape', '120'))
        self.assertEqual(args['--dataset.root'], self.env['WHITE_TAPE_DATASET_ROOT'])
        self.assertEqual(args['--dataset.repo_id'], 'local/white_tape')
        self.assertEqual(args['--dataset.single_task'], '抓取双面胶')
        self.assertEqual(args['--dataset.num_episodes'], '1')
        self.assertEqual(args['--dataset.episode_time_s'], '120')
        self.assertEqual(args['--resume'], 'false')
        self.assertEqual(len(json.loads(args['--robot.cameras'])), 3)
        self.assertFalse(Path(args['--dataset.root']).exists())

        alias = self.args_from(self.run_mode('demo-double-sided-tape', '120'))
        self.assertEqual(alias, args)

    def test_white_tape_resume_and_incomplete_directory(self):
        root = self.root / 'white_tape'
        root.mkdir()
        self.assertNotEqual(self.run_mode('demo-white-tape').returncode, 0)
        (root / 'meta').mkdir()
        (root / 'meta/info.json').write_text('{}')
        self.assertEqual(self.args_from(self.run_mode('demo-white-tape'))['--resume'], 'true')
        self.assertNotEqual(self.run_mode('demo-white-tape', '0').returncode, 0)

    def test_existing_metadata_selects_resume(self):
        meta = self.root / 'screwdriver/meta'
        meta.mkdir(parents=True)
        (meta / 'info.json').write_text('{}')
        self.assertEqual(self.args_from(self.run_mode('demo-screwdriver'))['--resume'], 'true')

    def test_clear_tape_separate_one_episode(self):
        args = self.args_from(self.run_mode('demo-clear-tape', '120'))
        self.assertEqual(args['--dataset.root'], self.env['CLEAR_TAPE_DATASET_ROOT'])
        self.assertEqual(args['--dataset.repo_id'], 'local/clear_tape')
        self.assertEqual(args['--dataset.single_task'], 'Pick up the clear tape.')
        self.assertEqual(args['--dataset.num_episodes'], '1')
        self.assertEqual(args['--dataset.episode_time_s'], '120')
        self.assertEqual(args['--resume'], 'false')
        self.assertEqual(len(json.loads(args['--robot.cameras'])), 3)
        self.assertFalse(Path(args['--dataset.root']).exists())

    def test_clear_tape_resume_and_incomplete_directory(self):
        root = self.root / 'clear_tape'
        root.mkdir()
        self.assertNotEqual(self.run_mode('demo-clear-tape').returncode, 0)
        (root / 'meta').mkdir()
        (root / 'meta/info.json').write_text('{}')
        self.assertEqual(self.args_from(self.run_mode('demo-clear-tape'))['--resume'], 'true')
        self.assertNotEqual(self.run_mode('demo-clear-tape', '0').returncode, 0)

    def test_existing_incomplete_root_and_bad_duration_rejected(self):
        (self.root / 'screwdriver').mkdir()
        self.assertNotEqual(self.run_mode('demo-screwdriver').returncode, 0)
        self.assertNotEqual(self.run_mode('demo-screwdriver', '0').returncode, 0)


if __name__ == '__main__':
    unittest.main()
