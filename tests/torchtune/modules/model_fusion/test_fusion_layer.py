# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.modules.model_fusion import FusionLayer
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class DummyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.cache_enabled = False

    def setup_cache(self, batch_size, dtype):
        self.cache_enabled = True

    def reset_cache(self):
        self.cache_enabled = False

    def forward(self, x):
        return self.linear(x)


class TestFusionLayer:
    """
    Class for testing our FusionLayer wrapper.
    """

    @pytest.fixture
    def dim(self) -> int:
        return 2

    @pytest.fixture
    def layer(self, dim) -> nn.Module:
        layer = DummyLayer(dim)
        fixed_init_model(layer, min_val=-0.1, max_val=0.1)
        return layer

    @pytest.fixture
    def fusion_layer(self, dim) -> nn.Module:
        layer = DummyLayer(dim)
        fixed_init_model(layer, min_val=-0.2, max_val=0.2)
        return layer

    @pytest.fixture
    def fused_layer(self, layer, fusion_layer) -> FusionLayer:
        return FusionLayer(layer, fusion_layer)

    @torch.no_grad()
    def test_forward(self, fused_layer, dim):
        """
        Test that the forward pass of the FusionLayer works as expected.
        """
        x = torch.rand((1, dim))
        out = fused_layer(x)

        assert out.shape == (1, dim)
        assert_expected(out.mean(), torch.tensor(-0.0316), atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_fusion_last_forward(self, layer, fusion_layer, dim) -> nn.Module:
        """
        Test the forward method with fusion_first=False.
        """
        fused_layer = FusionLayer(layer, fusion_layer, fusion_first=False)

        x = torch.rand((1, dim))
        out = fused_layer(x)

        assert out.shape == (1, dim)
        assert_expected(out.mean(), torch.tensor(-0.0816), atol=1e-3, rtol=1e-3)

    def test_get_and_load_state_dict(self, fused_layer):
        """
        Test that the state dict hooks work in removing the "layer" variable
        """
        state_dict = fused_layer.state_dict()
        state_keys = set(state_dict.keys())

        assert state_keys == {
            "linear.weight",
            "linear.bias",
            "fusion_layer.linear.weight",
            "fusion_layer.linear.bias",
        }

        # Check that the state_dict can be loaded back into the model
        fused_layer.load_state_dict(state_dict)

    def test_fusion_params(self, fused_layer):
        """
        Test that the currect fusion params are returned.
        """
        fusion_params = set(fused_layer.fusion_params())

        assert fusion_params == {
            "fusion_layer.linear.weight",
            "fusion_layer.linear.bias",
        }

    def test_setup_cache(self, fused_layer):
        """
        Test that the cache methods works as expected.
        """
        fused_layer.setup_cache(2, torch.float32)
        assert fused_layer.cache_enabled
        fused_layer.reset_cache()
        assert not fused_layer.cache_enabled
