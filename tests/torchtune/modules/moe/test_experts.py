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
    def num_tokens_per_expert(self, num_experts) -> int:
        return torch.tensor([1, 2, 1, 4, 3, 1, 2, 2], dtype=torch.int)

    @pytest.fixture
    def experts(self, dim, hidden_dim, num_experts) -> nn.Module:
        experts = GroupedExperts(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
        )
        fixed_init_model(experts, min_val=-0.1, max_val=0.1)
        return experts

    @torch.no_grad()
    def test_forward(self, experts, num_tokens_per_expert, dim):
        """
        Test that the forward pass of the experts works as expected.
        """
        x = torch.randn((16, dim))
        out = experts(x, num_tokens_per_expert)

        assert out.shape == (16, dim)
        assert_expected(out.mean().item(), 120.8260, atol=1e-3, rtol=1e-3)
