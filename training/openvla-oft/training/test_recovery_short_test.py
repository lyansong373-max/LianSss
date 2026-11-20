"""CPU-only tests for short-training schedules and full-validation aggregation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import recovery_short_test as trial


class RecoveryDiagnosticTests(unittest.TestCase):
    def test_warmup(self):
        self.assertAlmostEqual(trial.lr_at_step(1, 5e-5, 100), 5e-7)
        self.assertEqual(trial.lr_at_step(100, 5e-5, 100), 5e-5)
        self.assertEqual(trial.lr_at_step(1000, 5e-5, 100), 5e-5)
        self.assertEqual(trial.lr_at_step(1, 5e-5, 0), 5e-5)

    def test_full_training_constant_uses_training_actions(self):
        class Dataset:
            meta = SimpleNamespace(camera_keys=[])
            hf_dataset = {"action": [torch.full((16,), x) for x in (-1.0, 0.25, 1.0)]}

            def __getitem__(self, index):
                return {"action": torch.zeros(8, 16)}

        value = trial.full_training_constant(
            Dataset(), lambda s: {"action": s["action"].unsqueeze(0)}, SimpleNamespace(chunk_size=8))
        self.assertEqual(value.shape, (8, 16))
        self.assertTrue(torch.equal(value, torch.full((8, 16), 0.25)))

    def test_streaming_metrics_padding_and_smoke_scope(self):
        size = 65  # Two full 32-frame groups and one final partial group.
        target = torch.linspace(-1, 1, size)[:, None, None].expand(size, 8, 16).clone()
        prediction = target + 0.02
        padding = torch.zeros(size, 8, dtype=torch.bool)
        padding[-1, -3:] = True

        class Dataset:
            def __len__(self):
                return size

        def fake_cache(_dataset, indices, _pre, _config):
            indices = list(indices)
            records = [{"dataset_index": i, "episode": i // 33, "frame": i % 33,
                        "task": str(i // 33)} for i in indices]
            return {"action": target[indices], "action_is_pad": padding[indices],
                    "test_ids": torch.tensor(indices)}, records

        def fake_evaluate(_base, cached, constant, post, _batch_size):
            output = prediction[cached["test_ids"]]
            return trial.prediction_metrics(output, cached["action"], constant, post, 0.25), output

        with tempfile.TemporaryDirectory(prefix="oft-validation-unit-") as directory:
            root = Path(directory)
            with patch.object(trial, "cache_samples", fake_cache), patch.object(trial, "evaluate", fake_evaluate):
                full = trial.evaluate_full_dataset(
                    SimpleNamespace(config=None), Dataset(), None, lambda x: x,
                    torch.zeros(8, 16), torch.zeros(8, 16), 2, root, root / "events.jsonl")
                smoke = trial.evaluate_full_dataset(
                    SimpleNamespace(config=None), Dataset(), None, lambda x: x,
                    torch.zeros(8, 16), torch.zeros(8, 16), 2, root, root / "events.jsonl", limit=4)
            self.assertEqual(full["samples"], size)
            self.assertAlmostEqual(full["normalized_l1"], 0.02, places=6)
            self.assertAlmostEqual(full["valid_steps_normalized_l1"], 0.02, places=6)
            self.assertAlmostEqual(full["head_relu_zero_fraction"], 0.25)
            self.assertEqual([v["samples"] for v in full["per_task"].values()], [33, 32])
            self.assertTrue(full["beats_full_training_median_by_5pct"])
            self.assertEqual(smoke["samples"], 4)
            self.assertTrue((root / "full_validation_predictions.safetensors").is_file())
            self.assertEqual(json.loads((root / "full_validation.json").read_text())["samples"], size)
            self.assertEqual(json.loads((root / "full_validation_smoke.json").read_text())["samples"], 4)
            self.assertEqual(json.loads((root / "events.jsonl").read_text().splitlines()[-1])["event"],
                             "full_validation_smoke_complete")


if __name__ == "__main__":
    unittest.main()
