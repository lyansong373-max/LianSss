"""CPU checks using real PEFT and tiny FiLM modules; no robot or 7B allocation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import PretrainedConfig

import train_openvla_oft_fp32 as trainer
from lerobot_policy_openvla_oft.vision_backbone import FiLMGenerator
from lerobot_policy_openvla_oft.language_model import BidirectionalLlama, openvla_llama_config


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.use_film = True
        self.linear = torch.nn.Linear(4, 4)
        self.model = torch.nn.Module()
        self.model.action_head = torch.nn.Module()
        self.model.action_head.input_proj = torch.nn.Linear(4, 4)
        self.model.proprio_projector = torch.nn.Linear(4, 4)
        self.model.vision = torch.nn.Module()
        for name in ("dinov2", "siglip"):
            encoder = torch.nn.Module()
            encoder.film = FiLMGenerator(4, 2, 3)
            setattr(self.model.vision, name, encoder)

    def forward(self, x):
        output = self.model.action_head.input_proj(self.linear(x))
        output = output + self.model.proprio_projector(x)
        for name in ("dinov2", "siglip"):
            gamma, beta = getattr(self.model.vision, name).film(x)
            output = output + gamma.mean() + beta.mean()
        return output


class FilmTrainingTests(unittest.TestCase):
    def test_chunk_configuration_is_consistent_and_resume_cannot_change_it(self):
        cfg = SimpleNamespace(chunk_size=8, n_action_steps=8)
        trainer.configure_chunk(cfg, 30)
        self.assertEqual((cfg.chunk_size, cfg.n_action_steps), (30, 30))
        trainer.configure_chunk(cfg, 30, resume=True)
        with self.assertRaisesRegex(ValueError, "Cannot change action chunk"):
            trainer.configure_chunk(cfg, 8, resume=True)
        with self.assertRaises(ValueError):
            trainer.configure_chunk(cfg, 0)
        with self.assertRaises(ValueError):
            trainer.configure_chunk(cfg, 51)

    def test_checkpointing_preserves_lora_gradients_without_input_grad(self):
        torch.manual_seed(37)
        llm = BidirectionalLlama(openvla_llama_config(hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=4,
                                num_key_value_heads=4, vocab_size=32, pad_token_id=0))
        policy = get_peft_model(llm, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                               target_modules=["q_proj", "v_proj"]))
        base = SimpleNamespace(model=SimpleNamespace(llm=llm))
        inputs = torch.randn(2, 8, 16)  # Deliberately no requires_grad: frozen inputs.
        padding = torch.ones(2, 8, dtype=torch.bool)
        padding[0, -1] = False
        target = torch.randn_like(inputs)
        trainable = {name: p for name, p in policy.named_parameters() if p.requires_grad}
        keys_before = set(policy.state_dict())
        policy.train()
        trainer.configure_training_memory(base, False)
        reference = policy(inputs, padding)
        (reference - target).square().mean().backward()
        expected = {name: p.grad.clone() for name, p in trainable.items()}
        policy.zero_grad(set_to_none=True)
        settings = trainer.configure_training_memory(base, True)
        actual = policy(inputs, padding)
        (actual - target).square().mean().backward()
        self.assertTrue(settings["llm_gradient_checkpointing"])
        self.assertFalse(settings["llm_use_cache"])
        torch.testing.assert_close(reference, actual)
        self.assertTrue(any(p.grad.abs().sum() > 0 for p in trainable.values()))
        for name, parameter in trainable.items():
            torch.testing.assert_close(expected[name], parameter.grad)
        self.assertEqual(keys_before, set(policy.state_dict()))
        trainer.configure_training_memory(base, False)
        self.assertFalse(llm.model.is_gradient_checkpointing)

    def test_resume_cannot_change_film_architecture(self):
        cfg = SimpleNamespace(use_film=False, film_mask_padding=False)
        with self.assertRaises(ValueError):
            trainer.configure_film(cfg, use_film=True, resume=True)
        trainer.configure_film(cfg, use_film=True, mask_padding=True)
        self.assertTrue(cfg.use_film and cfg.film_mask_padding)
        trainer.configure_film(cfg, use_film=True, mask_padding=True, resume=True)

    def test_task_swap_preserves_images_state_and_changes_every_task(self):
        cache = {trainer.OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2], [1, 2], [1, 3], [1, 4]]),
                 trainer.OBS_LANGUAGE_ATTENTION_MASK: torch.ones(4, 2, dtype=torch.bool),
                 "observation.state": torch.randn(4, 16), "image": torch.randn(4, 3, 2, 2)}
        records = [{"task": task} for task in ("a", "a", "b", "c")]
        swapped = trainer.swap_probe_language(cache, records)
        self.assertIs(cache["image"], swapped["image"])
        self.assertIs(cache["observation.state"], swapped["observation.state"])
        self.assertEqual(swapped[trainer.OBS_LANGUAGE_TOKENS][:, 1].tolist(), [3, 3, 4, 2])
        self.assertEqual(cache[trainer.OBS_LANGUAGE_TOKENS][:, 1].tolist(), [2, 2, 3, 4])

    def test_real_peft_film_gradient_and_fp32_reload(self):
        torch.manual_seed(19)
        base = TinyBase()
        policy = get_peft_model(base, LoraConfig(r=2, target_modules=["linear"],
                               modules_to_save=["action_head", "proprio_projector", "film"]))
        trainer.normalize_trainable_precision(policy, base)
        parameters = trainer.film_trainable_parameters(base)
        self.assertEqual(len(parameters), 8)
        policy(torch.randn(2, 4)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            and p.grad.abs().sum() > 0 for p in parameters.values()))
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=5e-5)
        optimizer.step()
        # Deliberately use values that would change if rounded through BF16.
        with torch.no_grad():
            for p in parameters.values():
                p.copy_(torch.linspace(0.1001, 0.1019, p.numel()).reshape_as(p))
        expected = {name: p.detach().clone() for name, p in parameters.items()}
        with tempfile.TemporaryDirectory(prefix="oft-film-test-") as directory:
            root = Path(directory)
            policy.save_pretrained(root, safe_serialization=True)
            precision = trainer.verify_checkpoint_precision(root, True)
            self.assertEqual(precision["film_tensor_count"], 8)
            (root / "precision_requirements.json").write_text(json.dumps(precision))
            with patch.object(trainer.OpenVLAOFTConfig, "from_pretrained", return_value=base.config), \
                 patch.object(trainer.OpenVLAOFTPolicy, "from_pretrained",
                              side_effect=lambda *a, **kw: TinyBase().bfloat16()), \
                 patch.object(trainer.PeftModel, "from_pretrained",
                              wraps=trainer.PeftModel.from_pretrained) as adapter_loader:
                loaded, loaded_base = trainer.load_fp32_adapter(root, root, root)
                self.assertEqual(adapter_loader.call_args.kwargs["torch_device"], "cpu")
            for name, module in trainer.fp32_modules(loaded_base).items():
                if name.endswith(".film"):
                    for suffix, p in module.modules_to_save["default"].named_parameters():
                        self.assertEqual(p.dtype, torch.float32)
                        self.assertTrue(torch.equal(p, expected[f"{name}.{suffix}"]))
            tensors = load_file(root / "adapter_model.safetensors")
            tensors.pop(next(k for k in tensors if ".film." in k))
            save_file(tensors, root / "adapter_model.safetensors")
            with self.assertRaises(RuntimeError):
                trainer.verify_checkpoint_precision(root, True)


if __name__ == "__main__":
    unittest.main()
