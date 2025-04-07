# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.modules.moe import GroupedExperts
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class TestExperts:
    """
    Class for testing our Experts module.
    """

    @pytest.fixture
    def dim(self) -> int:
        return 1280

    @pytest.fixture
    def hidden_dim(self) -> int:
        return 8192

    @pytest.fixture
    def num_experts(self) -> int:
        return 8

    @pytest.fixture
    def experts(self, dim, hidden_dim, num_experts) -> nn.Module:
        experts = GroupedExperts(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
        )
        fixed_init_model(experts, min_val=-0.1, max_val=0.1)
        return experts

    @torch.no_grad()
    def test_forward(self, experts, dim):
        """
        Test that the forward pass of the experts works as expected.
        """
        x = torch.randn((16, dim)).view(8, 2, dim)
        out = experts(x)

        assert out.shape == (8, 2, dim)
        assert_expected(out.mean(), torch.tensor(18.5488), atol=1e-3, rtol=1e-3)

    def test_get_and_load_state_dict(self, experts):
        """
        Test that the state dict hooks work in removing the "layer" variable
        """
        state_dict = experts.state_dict()
        state_keys = set(state_dict.keys())

        assert state_keys == {
            "gate_proj",
            "down_proj",
            "up_proj",
        }

        # Check that the state_dict can be loaded back into the model
        experts.load_state_dict(state_dict)
