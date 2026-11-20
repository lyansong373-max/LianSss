"""CPU tests for full-dataset sampling, preprocessing and resumable state."""
import json
import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

import train_openvla_oft_fp32 as trainer


class FullTrainingTests(unittest.TestCase):
    def test_15000_step_cli_and_eta(self):
        with tempfile.TemporaryDirectory(prefix="oft-args-test-") as directory:
            inputs = [item for key in ("base", "reference", "data", "tokenizer", "output")
                      for item in (f"--{key}", directory)]
            with patch("sys.argv", ["trainer", *inputs, "--steps", "15000"]):
                self.assertEqual(trainer.parse_args().steps, 15000)
            for invalid in ("0", "15001"):
                with patch("sys.argv", ["trainer", *inputs, "--steps", invalid]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    trainer.parse_args()
        progress = trainer.training_progress(5000, 15000, 4000)
        self.assertEqual(progress["remaining_steps"], 10000)
        self.assertEqual(progress["remaining_s"], 8000)
        self.assertEqual(trainer.training_progress(15000, 15000, 12000)["remaining_s"], 0)

    def test_skipped_validation_is_never_reported_as_passed(self):
        summary = trainer.unvalidated_training_summary(5000, Path("checkpoints/005000"), 10000)
        self.assertEqual(summary["step"], 5000)
        self.assertEqual(summary["unique_training_frames_seen"], 10000)
        self.assertEqual(summary["offline_validation"], "skipped_by_user")
        self.assertIsNone(summary["all_offline_checks_pass"])
        self.assertFalse(summary["robot_deployment_approved"])

    def test_train_only_augmentation_preserves_labels_and_cameras(self):
        torch.manual_seed(7)
        image = torch.rand(2, 3, 8, 8)
        raw = {"camera1": image, "camera2": image.clone(),
               "observation.state": torch.rand(2, 16), "action": torch.rand(2, 8, 16),
               trainer.OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
               trainer.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool)}
        cfg = SimpleNamespace(image_features={"camera1": None, "camera2": None}, chunk_size=8)
        clean = trainer.process_batch(raw, lambda x: x, cfg, device="cpu")
        augmented = trainer.process_batch(raw, lambda x: x, cfg, device="cpu", image_augmentation=True)
        self.assertTrue(torch.equal(clean["camera1"], image))
        self.assertFalse(torch.equal(augmented["camera1"], image))
        self.assertTrue(torch.equal(augmented["camera1"], augmented["camera2"]))
        self.assertTrue(0 <= augmented["camera1"].min() <= augmented["camera1"].max() <= 1)
        self.assertTrue(torch.equal(raw["camera1"], image))
        for key in ("observation.state", "action", trainer.OBS_LANGUAGE_TOKENS):
            self.assertTrue(torch.equal(clean[key], augmented[key]))

    def test_progress_eta_including_resume(self):
        progress = trainer.training_progress(100, 5000, 80)
        self.assertEqual(progress["progress_percent"], 2.0)
        self.assertEqual(progress["remaining_time"], "01:05:20")
        resumed = trainer.training_progress(2600, 5000, 80, start_step=2500)
        self.assertEqual(resumed["remaining_s"], 1920)
        self.assertEqual(trainer.training_progress(5000, 5000, 4000)["remaining_time"], "00:00:00")

    def test_sampler_resume_matches_uninterrupted_order(self):
        batches = list(trainer.StepBatchSampler(11, 2, 13, 1000))
        resumed = list(trainer.StepBatchSampler(11, 2, 13, 1000, start_step=7))
        self.assertEqual(len(batches), 13)
        self.assertEqual(resumed, batches[7:])
        for start in (0, 5):
            self.assertEqual(len({i for row in batches[start:start + 5] for i in row}), 10)
        self.assertEqual(list(trainer.StepBatchSampler(11, 2, 13, 1000, start_step=13)), [])

    def test_preprocessing_uint8_and_shape(self):
        config = SimpleNamespace(image_features={"observation.images.camera1": None}, chunk_size=8)
        raw = {"observation.images.camera1": torch.full((2, 3, 4, 4), 255, dtype=torch.uint8),
               "observation.state": torch.zeros(2, 16), "action": torch.zeros(2, 8, 16),
               trainer.OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
               trainer.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool)}
        result = trainer.process_batch(raw, lambda x: x, config, device="cpu")
        self.assertTrue((result["observation.images.camera1"] == 1).all())
        self.assertEqual(raw["observation.images.camera1"].dtype, torch.uint8)
        raw["action"] = torch.zeros(2, 7, 16)
        with self.assertRaises(ValueError):
            trainer.process_batch(raw, lambda x: x, config, device="cpu")

    def test_rng_and_weights_only_optimizer_roundtrip(self):
        model = torch.nn.Linear(3, 2).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, foreach=False)
        model(torch.ones(2, 3)).square().mean().backward()
        optimizer.step()
        with patch.object(torch.cuda, "is_available", return_value=False):
            state = trainer.rng_state()
            expected = (random.random(), float(np.random.random()), torch.rand(2))
            trainer.restore_rng(state)
            actual = (random.random(), float(np.random.random()), torch.rand(2))
        self.assertEqual(actual[0:2], expected[0:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        with tempfile.TemporaryDirectory(prefix="oft-state-test-") as directory:
            path = Path(directory) / "state.pt"
            torch.save({"optimizer": trainer.cpu_tree(optimizer.state_dict()), "rng": state}, path)
            loaded = torch.load(path, weights_only=True, map_location="cpu")
            optimizer.load_state_dict(loaded["optimizer"])
        self.assertTrue(all(s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32
                            for s in optimizer.state.values()))

    def test_checkpoint_transaction_and_no_overwrite(self):
        class ToyPolicy(torch.nn.Linear):
            def save_pretrained(self, path, **kwargs):
                save_file({"base_model.model.model.action_head.input_proj.weight": self.weight.detach()},
                          str(path / "adapter_model.safetensors"))

        class Config:
            def save_pretrained(self, path):
                (path / "config.json").write_text("{}")

        with tempfile.TemporaryDirectory(prefix="oft-checkpoint-test-") as directory:
            output = Path(directory)
            (output / "run_config.json").write_text("{}")
            policy = ToyPolicy(3, 2)
            optimizer = torch.optim.AdamW(policy.parameters())
            config = Config()
            config.policy = config
            args = SimpleNamespace(output=output)
            with patch.object(trainer.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 2**30)), \
                 patch.object(torch.cuda, "is_available", return_value=False):
                path = trainer.save_checkpoint(policy, config, config, config, optimizer, 10,
                                               args, output / "metrics.jsonl")
                with self.assertRaises(FileExistsError):
                    trainer.save_checkpoint(policy, config, config, config, optimizer, 10,
                                            args, output / "metrics.jsonl")
            self.assertTrue((path / "complete.json").is_file())
            self.assertFalse((output / "checkpoints/.incomplete-000010").exists())
            self.assertEqual(torch.load(path / "training_state.pt", weights_only=True)["step"], 10)
            self.assertEqual(json.loads((output / "latest.json").read_text())["step"], 10)


if __name__ == "__main__":
    unittest.main()
