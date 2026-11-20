"""CPU-only tests: augmentation units, labels, reproducibility and clean defaults."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig

import train_openvla_oft_fp32 as trainer


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        image = torch.rand(4, 3, 48, 64)
        self.raw = {"camera1": image, "camera2": image.clone(), "camera3": image.clone(),
                    "observation.state": torch.rand(4, 16), "action": torch.rand(4, 8, 16),
                    trainer.OBS_LANGUAGE_TOKENS: torch.ones(4, 6, dtype=torch.long),
                    trainer.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(4, 6, dtype=torch.bool)}
        self.cfg = SimpleNamespace(image_features={f"camera{i}": None for i in range(1, 4)}, chunk_size=8)

    def process(self, **options):
        return trainer.process_batch(self.raw, lambda x: x, self.cfg, device="cpu", **options)

    def test_all_defaults_are_clean_and_do_not_consume_rng(self):
        rng = torch.get_rng_state()
        batch = self.process()
        batch, keep = trainer.augment_training_state(batch, None, trainer.AUGMENTATION_DEFAULTS)
        self.assertIsNone(keep)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key in self.raw:
            self.assertTrue(torch.equal(batch[key], self.raw[key]), key)

    def test_noise_is_scaled_after_normalization_and_labels_source_unchanged(self):
        original = copy.deepcopy(self.raw)
        ranges = torch.linspace(0.51, 1.72, 16)
        options = {**trainer.AUGMENTATION_DEFAULTS, "state_noise_std_rad": 0.03}
        torch.manual_seed(33)
        physical_noise = (torch.randn_like(self.raw["observation.state"]) * 0.03).clamp(-0.09, 0.09)
        physical_noise[:, trainer.GRIPPER_STATE_INDICES] = 0
        expected = (self.raw["observation.state"] + 2 * physical_noise / ranges).clamp(-1, 1)
        torch.manual_seed(33)
        batch, keep = trainer.augment_training_state(self.process(), ranges, options)
        self.assertIsNone(keep)
        self.assertTrue(torch.equal(batch["observation.state"], expected))
        self.assertTrue(torch.equal(batch["action"], original["action"]))
        grippers = trainer.GRIPPER_STATE_INDICES
        self.assertTrue(torch.equal(batch["observation.state"][:, grippers], original["observation.state"][:, grippers]))
        for key in self.raw:
            self.assertTrue(torch.equal(self.raw[key], original[key]), key)

    def test_noise_distribution_and_three_sigma_cap(self):
        metrics = {}
        state = torch.zeros(50000, 16)
        ranges = torch.linspace(0.51, 1.72, 16)
        result = trainer.perturb_normalized_state(state, ranges, 0.03, 0.02, metrics=metrics)
        physical_noise = result * ranges / 2
        arms = physical_noise[:, trainer.ARM_STATE_INDICES]
        self.assertLess(abs(float(arms.mean())), 0.0003)
        self.assertAlmostEqual(float(arms.std()), 0.03, delta=0.0005)
        for sigma in arms.std(dim=0):
            self.assertAlmostEqual(float(sigma), 0.03, delta=0.0005)
        for sigma in physical_noise[:, trainer.GRIPPER_STATE_INDICES].std(dim=0):
            self.assertAlmostEqual(float(sigma), 0.02, delta=0.0005)
        self.assertLessEqual(float(arms.abs().max()), 0.090001)
        self.assertLessEqual(metrics["gripper_noise_max_abs"], 0.060001)
        self.assertEqual(int(torch.count_nonzero(state)), 0)

    def test_dropout_does_not_zero_input_and_noise_skips_dropped_rows(self):
        options = {**trainer.AUGMENTATION_DEFAULTS, "state_dropout_prob": 0.3}
        batch, keep = trainer.augment_training_state(self.process(), None, options)
        self.assertTrue(torch.equal(batch["observation.state"], self.raw["observation.state"]))
        state = torch.zeros(10000, 16)
        keep = trainer.sample_state_keep(state, 0.3)
        self.assertAlmostEqual(float((~keep).float().mean()), 0.3, delta=0.02)
        perturbed = trainer.perturb_normalized_state(state, torch.ones(16), 0.03, 0.02, keep)
        self.assertTrue(torch.equal(perturbed[~keep], state[~keep]))
        self.assertGreater(float(perturbed[keep].abs().sum()), 0)
        boundary = torch.ones(4, 16)
        boundary[2:] = -1
        result = trainer.perturb_normalized_state(boundary, torch.ones(16), 0.03, 0.02)
        self.assertTrue((result.abs() <= 1).all())

    def test_real_peft_projector_token_hook_gradients_cleanup_and_checkpoint_format(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = PretrainedConfig()
                self.linear = torch.nn.Linear(16, 16)
                self.proprio_projector = trainer.ProprioProjector(16, 8)

            def forward(self, state):
                return self.proprio_projector(self.linear(state))

        model = get_peft_model(Tiny(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                               target_modules=["linear"], modules_to_save=["proprio_projector"]))
        model.train()
        state = torch.rand(4, 16, requires_grad=True)
        projector = model.get_base_model().proprio_projector
        keys = set(model.state_dict())
        clean = model(state).detach()
        hook = trainer.ProprioTokenDropout(projector)
        keep = torch.tensor([True, False, True, False])
        try:
            with hook.masked(keep):
                output = model(state)
                self.assertTrue(torch.equal(output[keep], clean[keep]))
                self.assertTrue((output[~keep] == 0).all())
                output.sum().backward()
            self.assertIsNone(hook.keep)
            self.assertTrue((state.grad[~keep] == 0).all())
            self.assertGreater(float(state.grad[keep].abs().sum()), 0)
            model.zero_grad(set_to_none=True)
            with hook.masked(torch.zeros(4, dtype=torch.bool)):
                model(state).sum().backward()
            self.assertTrue(all((p.grad == 0).all() for p in projector.parameters() if p.grad is not None))
            self.assertTrue(torch.equal(model(state), clean))
            model.eval()
            with hook.masked(torch.zeros(4, dtype=torch.bool)):
                self.assertTrue(torch.equal(model(state), clean))
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with hook.masked(keep):
                    raise RuntimeError("test failure")
            self.assertIsNone(hook.keep)
            self.assertEqual(keys, set(model.state_dict()))
        finally:
            hook.close()
        self.assertEqual(len(projector._forward_hooks), 0)
        self.assertTrue(torch.equal(model(state), clean))

    def test_actual_normalizer_statistics_must_match(self):
        from lerobot.configs.types import PolicyFeature
        stats = {"observation.state": {"min": [-1.0] * 16, "max": [1.0] * 16}}
        normalizer = trainer.NormalizerProcessorStep(
            features={"observation.state": PolicyFeature(type=trainer.FeatureType.STATE, shape=(16,))},
            norm_map={trainer.FeatureType.STATE: trainer.NormalizationMode.MIN_MAX}, stats=stats)
        pre = SimpleNamespace(steps=[normalizer])
        self.assertTrue(torch.equal(trainer.state_range_from_preprocessor(pre, stats), torch.full((16,), 2.0)))
        bad = copy.deepcopy(stats)
        bad["observation.state"]["max"][0] += 0.1
        with self.assertRaises(ValueError):
            trainer.state_range_from_preprocessor(pre, bad)
        normalizer.norm_map[trainer.FeatureType.STATE] = trainer.NormalizationMode.MEAN_STD
        with self.assertRaises(ValueError):
            trainer.state_range_from_preprocessor(pre, stats)

    def test_crop_and_color_preserve_labels_source_and_share_camera_parameters(self):
        original = copy.deepcopy(self.raw)
        with patch.object(trainer.image_functional, "resized_crop",
                          wraps=trainer.image_functional.resized_crop) as crop:
            batch = self.process(random_crop=True, image_augmentation=True)
        calls = crop.call_args_list
        self.assertEqual(len(calls), 12)
        for i in range(4):
            # Identical relative crop for the three same-sized camera images.
            self.assertEqual(calls[i].args[1:], calls[i + 4].args[1:])
            self.assertEqual(calls[i].args[1:], calls[i + 8].args[1:])
            _, top, left, height, width, output_size = calls[i].args
            self.assertGreaterEqual(height * width / (48 * 64), 0.88)  # pixel rounding
            self.assertLessEqual(top + height, 48)
            self.assertLessEqual(left + width, 64)
            self.assertEqual(output_size, [48, 64])
        self.assertFalse(torch.equal(batch["camera1"], original["camera1"]))
        self.assertTrue(torch.equal(batch["camera1"], batch["camera2"]))
        self.assertTrue(torch.equal(batch["camera1"], batch["camera3"]))
        self.assertTrue(0 <= batch["camera1"].min() <= batch["camera1"].max() <= 1)
        for key in original:
            self.assertTrue(torch.equal(self.raw[key], original[key]), key)
            if key not in self.cfg.image_features:
                self.assertTrue(torch.equal(batch[key], original[key]), key)

    def test_crop_and_color_independent_switches(self):
        with patch.object(trainer.image_functional, "adjust_brightness") as color:
            cropped = self.process(random_crop=True)
            color.assert_not_called()
        self.assertFalse(torch.equal(cropped["camera1"], self.raw["camera1"]))
        full = self.process(random_crop=True, random_crop_min_scale=1)
        torch.testing.assert_close(full["camera1"], self.raw["camera1"], rtol=0, atol=0)
        with patch.object(trainer.image_functional, "resized_crop") as crop:
            self.process(image_augmentation=True)
            crop.assert_not_called()

    def test_rng_restore_reproduces_all_augmentations_and_clean_batch_stays_clean(self):
        options = {**trainer.AUGMENTATION_DEFAULTS, "state_noise_std_rad": 0.03,
                   "state_noise_gripper": 0.02, "state_dropout_prob": 0.3}
        with patch.object(torch.cuda, "is_available", return_value=False):
            rng = trainer.rng_state()
            first, first_keep = trainer.augment_training_state(
                self.process(random_crop=True, image_augmentation=True), torch.ones(16), options)
            trainer.restore_rng(rng)
            second, second_keep = trainer.augment_training_state(
                self.process(random_crop=True, image_augmentation=True), torch.ones(16), options)
        self.assertTrue(torch.equal(first_keep, second_keep))
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        clean = self.process()
        for key in self.raw:
            self.assertTrue(torch.equal(clean[key], self.raw[key]), key)

    def test_invalid_values_and_state_layout_rejected(self):
        defaults = dict(trainer.AUGMENTATION_DEFAULTS)
        trainer.validate_augmentation(defaults)
        for key, values in {"state_noise_std_rad": [-0.01, 0.051, float("nan"), float("inf")],
                            "state_noise_gripper": [-0.01, 0.051, float("nan")],
                            "state_dropout_prob": [-0.1, 0.51, float("nan")],
                            "random_crop_min_scale": [0, -0.1, 1.1, float("nan")]}.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    trainer.validate_augmentation({**defaults, key: value})
        feature = {"shape": [16], "names": list(trainer.EXPECTED_STATE_NAMES)}
        trainer.validate_state_layout({"observation.state": feature})
        feature["names"][7], feature["names"][8] = feature["names"][8], feature["names"][7]
        with self.assertRaises(ValueError):
            trainer.validate_state_layout({"observation.state": feature})

    def test_resume_rejects_changed_augmentation_but_accepts_legacy_disabled(self):
        defaults = dict(trainer.AUGMENTATION_DEFAULTS)
        trainer.validate_resume_augmentation(defaults, {})
        for key, value in {"state_noise_std_rad": 0.03, "state_noise_gripper": 0.02, "state_dropout_prob": 0.3,
                           "image_augmentation": True, "random_crop": True,
                           "random_crop_min_scale": 0.95}.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                trainer.validate_resume_augmentation({**defaults, key: value}, {})


if __name__ == "__main__":
    unittest.main()
