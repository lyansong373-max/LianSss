#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, CPU-only checks for PI0's optional FiLM and state perturbations."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from lerobot.policies.pi0.modeling_pi0 import (  # noqa: E402
    LanguageConditionedFiLM,
    PI0Pytorch,
    drop_projected_state,
    normalized_state_noise_std,
)


class _Prefix(nn.Module):
    embed_prefix = PI0Pytorch.embed_prefix

    def __init__(self):
        super().__init__()
        self.language_embedding = nn.Embedding(20, 4)
        self.image_film = LanguageConditionedFiLM(4, 3)
        self.paligemma_with_expert = SimpleNamespace(embed_language_tokens=self.embed_language_tokens)

    def embed_language_tokens(self, tokens):
        return self.language_embedding(tokens)

    def _apply_checkpoint(self, func, *args):
        return func(*args)

    def _embed_images(self, images):
        return images


def test_film_starts_as_identity_and_uses_instruction():
    prefix = _Prefix()
    images = [torch.randn(1, 3, 4).expand(2, -1, -1).clone()]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    tokens = torch.tensor([[1, 2], [3, 4]])
    token_mask = torch.ones_like(tokens, dtype=torch.bool)

    original, _, _ = prefix.embed_prefix(images, image_masks, tokens, token_mask)
    torch.testing.assert_close(original[:, :3], images[0], rtol=0, atol=0)

    with torch.no_grad():
        prefix.language_embedding.weight.zero_()
        prefix.language_embedding.weight[1:3].fill_(1.0)
        prefix.language_embedding.weight[3:5].fill_(2.0)
        prefix.image_film.generator[0].weight.fill_(0.25)
        prefix.image_film.generator[0].bias.zero_()
        prefix.image_film.generator[-1].weight.fill_(0.1)
    changed, _, _ = prefix.embed_prefix(images, image_masks, tokens, token_mask)
    assert not torch.equal(changed[:, :3], original[:, :3])
    assert not torch.equal(changed[0, :3], changed[1, :3])


def test_state_dropout_is_training_only_and_approximately_fifteen_percent():
    state = torch.ones(10_000, 4)
    torch.testing.assert_close(drop_projected_state(state, 0.15, False), state, rtol=0, atol=0)
    torch.testing.assert_close(drop_projected_state(state, 0.0, True), state, rtol=0, atol=0)

    torch.manual_seed(0)
    dropped = drop_projected_state(state, 0.15, True)
    assert ((dropped == 0).all(dim=1).float().mean() - 0.15).abs() < 0.02
    assert ((dropped == 1).all(dim=1) | (dropped == 0).all(dim=1)).all()


def test_film_and_dropout_config_validation():
    config = PI0Config(
        use_film=True, state_dropout_prob=0.15, state_noise_rad=0.03, state_noise_gripper=0.02
    )
    assert config.use_film and config.state_dropout_prob == 0.15
    with pytest.raises(ValueError, match="state_dropout_prob"):
        PI0Config(state_dropout_prob=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        PI0Config(state_noise_rad=-0.01)


def test_state_noise_is_converted_from_raw_units_per_joint():
    state_std = torch.tensor([0.1, 0.2, 0.25])
    names = ["left_joint1.pos", "left_joint2.pos", "left_gripper.pos"]
    sigma = normalized_state_noise_std(state_std, names, 0.03, 0.02)
    torch.testing.assert_close(sigma * state_std, torch.tensor([0.03, 0.03, 0.02]))
