"""CPU-only checks for strict single-object selection; no robot/model access."""
import copy
import unittest
from pathlib import Path
from prepare_screwdriver_reference import TASK, make_config, select_episodes, verify_frame_rows


class ScrewdriverSelectionTests(unittest.TestCase):
    def records(self):
        return [{"episode_index": i, "tasks": [TASK if i < 10 else "other task"], "length": 600}
                for i in range(60)]

    def test_only_screwdriver_and_episode_disjoint_split(self):
        split = select_episodes(self.records()[::-1], 0.1)
        self.assertEqual(split["selected_episodes"], list(range(10)))
        self.assertEqual(split["train_episodes"], list(range(9)))
        self.assertEqual(split["validation_episodes"], [9])
        self.assertEqual((split["train_frames"], split["validation_frames"]), (5400, 600))

    def test_mixed_missing_duplicate_and_empty_are_rejected(self):
        for records in ([{"episode_index": 0, "tasks": [TASK, "other"], "length": 600}],
                        [{"episode_index": 0, "tasks": ["other"], "length": 600}],
                        self.records() + [self.records()[0]],
                        [{"episode_index": 0, "tasks": [TASK], "length": 0}]):
            with self.assertRaises(ValueError):
                select_episodes(records, 0.1)

    def test_bad_splits_rejected(self):
        for value in (0, 1, -0.1, 0.99):
            with self.assertRaises(ValueError):
                select_episodes(self.records(), value)

    def test_frame_task_mismatch_and_missing_frames_rejected(self):
        records = [{"episode_index": 0, "tasks": [TASK], "length": 2}]
        rows = [{"episode_index": 0, "task_index": 0, "frame_index": i} for i in range(2)]
        verify_frame_rows(rows, 0, records, [0])
        for bad in (rows[:1], rows + [rows[0]], [dict(rows[0], task_index=1), rows[1]]):
            with self.assertRaises(ValueError):
                verify_frame_rows(bad, 0, records, [0])

    def test_reference_is_not_mutated_or_resumed(self):
        config = {"dataset": {"episodes": None, "eval_split": 0.1,
                              "image_transforms": {"enable": False}}, "resume": True}
        original = copy.deepcopy(config)
        filtered = make_config(config, Path("/existing/data"), select_episodes(self.records(), 0.1))
        self.assertEqual(config, original)
        self.assertEqual(filtered["dataset"]["episodes"], list(range(10)))
        self.assertFalse(filtered["resume"])
        self.assertNotIn("exclude_episodes", filtered["dataset"])


if __name__ == "__main__":
    unittest.main()
