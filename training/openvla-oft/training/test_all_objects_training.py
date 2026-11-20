"""Validate all-task selection and reject accidentally inherited single-task filters."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from prepare_all_objects_reference import select_all_episodes, make_config
from train_openvla_oft_fp32 import verify_all_objects_split


class AllObjectsTests(unittest.TestCase):
    def records(self):
        return [{"episode_index": i, "tasks": [f"task{i // 10}"], "length": 600} for i in range(60)]

    def test_all_six_tasks_in_both_splits(self):
        split = select_all_episodes(self.records()[::-1], 0.1)
        self.assertEqual(split["selected_episodes"], list(range(60)))
        self.assertEqual(split["validation_episodes"], [9, 19, 29, 39, 49, 59])
        self.assertEqual(len(split["train_episodes"]), 54)
        self.assertEqual((split["train_frames"], split["validation_frames"]), (32400, 3600))
        self.assertEqual(len(split["per_task"]), 6)
        for item in split["per_task"].values():
            self.assertEqual((item["train_frames"], item["validation_frames"]), (5400, 600))

    def test_invalid_metadata_rejected(self):
        for records in (self.records()[:10], self.records()[:50], self.records() + [self.records()[0]],
                        [dict(self.records()[0], tasks=["a", "b"])] + self.records()[1:],
                        [dict(self.records()[0], length=0)] + self.records()[1:]):
            with self.assertRaises(ValueError):
                select_all_episodes(records, 0.1)
        for split in (0, 1, 0.99, float("nan")):
            with self.assertRaises(ValueError):
                select_all_episodes(self.records(), split)

    def test_inherited_screwdriver_filter_is_replaced_without_source_mutation(self):
        source = {"dataset": {"episodes": list(range(10)), "exclude_episodes": [11],
                              "eval_split": 0.1, "image_transforms": {"enable": True}}, "resume": True}
        original = copy.deepcopy(source)
        config = make_config(source, Path("/data"), select_all_episodes(self.records(), 0.1))
        self.assertEqual(source, original)
        self.assertEqual(config["dataset"]["episodes"], list(range(60)))
        self.assertIsNone(config["dataset"]["exclude_episodes"])
        self.assertFalse(config["resume"])

    def test_runtime_split_must_match_frame_tasks_and_episodes(self):
        class Dataset:
            def __init__(self, episodes):
                self.episodes = episodes
                self.hf_dataset = {"task_index": [ep // 10 for ep in episodes for _ in range(600)]}
            def __len__(self):
                return len(self.hf_dataset["task_index"])
        manifest = select_all_episodes(self.records(), 0.1)
        manifest["task_ids"] = {f"task{i}": i for i in range(6)}
        train, validation = Dataset(manifest["train_episodes"]), Dataset(manifest["validation_episodes"])
        counts = verify_all_objects_split(train, validation, manifest)
        self.assertEqual(len(counts["train"]), 6)
        train.hf_dataset["task_index"][0] = 99
        with self.assertRaises(ValueError):
            verify_all_objects_split(train, validation, manifest)
        with self.assertRaises(ValueError):
            verify_all_objects_split(Dataset(list(range(9))), validation, manifest)


if __name__ == "__main__":
    unittest.main()
