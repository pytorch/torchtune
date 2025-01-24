# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtune.models.flux._flow_model import FluxFlowModel
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
