# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch

from torchtune.models.flux._flow_model import FluxFlowModel
from torchtune.models.flux._model_builders import _replace_linear_with_lora
from torchtune.models.flux._util import predict_noise
from torchtune.training.seed import set_seed

BSZ = 32
CH = 4
RES = 8
Y_DIM = 16
TXT_DIM = 8

# model inputs/outputs are sequences of 2x2 latent patches
MODEL_CH = CH * 4


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestFluxFlowModel:
    @pytest.fixture
    def model(self):
        model = FluxFlowModel(
            in_channels=MODEL_CH,
            out_channels=MODEL_CH,
            vec_in_dim=Y_DIM,
            context_in_dim=TXT_DIM,
            hidden_size=16,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[2, 2, 4],
            theta=10_000,
            qkv_bias=True,
            use_guidance=True,
        )

        for param in model.parameters():
            param.data.uniform_(0, 0.1)

        return model

    @pytest.fixture
    def latents(self):
        return torch.randn(BSZ, CH, RES, RES)

    @pytest.fixture
    def clip_encodings(self):
        return torch.randn(BSZ, Y_DIM)

    @pytest.fixture
    def t5_encodings(self):
        return torch.randn(BSZ, 8, TXT_DIM)

    @pytest.fixture
    def timesteps(self):
        return torch.rand(BSZ)

    @pytest.fixture
    def guidance(self):
        return torch.rand(BSZ) * 3 + 1

    def test_forward(
        self, model, latents, clip_encodings, t5_encodings, timesteps, guidance
    ):
        actual = predict_noise(
            model, latents, clip_encodings, t5_encodings, timesteps, guidance
        )
        assert actual.shape == (BSZ, CH, RES, RES)

        actual = torch.mean(actual, dim=(0, 2, 3))
        print(actual)
        expected = torch.tensor([1.9532, 2.0414, 2.2768, 2.2754])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_backward(
        self, model, latents, clip_encodings, t5_encodings, timesteps, guidance
    ):
        pred = predict_noise(
            model, latents, clip_encodings, t5_encodings, timesteps, guidance
        )
        loss = pred.mean()
        loss.backward()

    def test_lora(
        self, model, latents, clip_encodings, t5_encodings, timesteps, guidance
    ):
        # Setup LoRA model
        lora_model = deepcopy(model)
        _replace_linear_with_lora(
            lora_model,
            rank=2,
            alpha=2,
            dropout=0.0,
            quantize_base=False,
            use_dora=False,
        )
        lora_model.load_state_dict(model.state_dict(), strict=False)
        lora_model.requires_grad_(False)
        _lora_enable_grad(lora_model)

        # Check param counts
        total_params = sum(p.numel() for p in lora_model.parameters())
        trainable_params = sum(
            p.numel() for p in lora_model.parameters() if p.requires_grad
        )
        assert total_params == 24416
        assert trainable_params == 3280

        # Check parity with original model
        pred = predict_noise(
            model, latents, clip_encodings, t5_encodings, timesteps, guidance
        )
        lora_pred = predict_noise(
            lora_model, latents, clip_encodings, t5_encodings, timesteps, guidance
        )
        torch.testing.assert_close(pred, lora_pred, atol=1e-4, rtol=1e-4)

        # Check backward pass works
        loss = lora_pred.mean()
        loss.backward()


def _lora_enable_grad(module):
    for name, child in module.named_children():
        if name.startswith("lora_"):
            child.requires_grad_(True)
        else:
            _lora_enable_grad(child)
