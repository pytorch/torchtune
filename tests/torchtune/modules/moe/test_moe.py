# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.modules.moe import ExpertChoiceTopKRouter, GroupedExperts, MoE
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class TestMoeLayer:
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
    def capacity_factor(self) -> float:
        return 1.0

    @pytest.fixture
    def experts(self, dim, hidden_dim, num_experts) -> int:
        return GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)

    @pytest.fixture
    def router(self, dim, num_experts, capacity_factor) -> int:
        return ExpertChoiceTopKRouter(
            gate=nn.Linear(dim, num_experts, bias=False),
            dim=dim,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

    @pytest.fixture
    def shared_expert(self, dim, hidden_dim) -> int:
        return GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=1)

    @pytest.fixture
    def moe(self, experts, router, shared_expert) -> nn.Module:
        moe = MoE(experts=experts, router=router, shared_expert=shared_expert)
        fixed_init_model(moe, min_val=-0.1, max_val=0.1)
        return moe

    @torch.no_grad()
    def test_forward(self, moe, dim):
        """
        Test that the forward pass of the moe layer works as expected.
        """
        x = torch.randn((16, dim)).view(2, 8, dim)
        out = moe(x)

        assert out.shape == (2, 8, dim)
        assert_expected(out.mean(), torch.tensor(215.6440), atol=1e-3, rtol=1e-3)

    def test_get_and_load_state_dict(self, moe):
        """
        Test that the state dict hooks work in removing the "layer" variable
        """
        state_dict = moe.state_dict()
        state_keys = set(state_dict.keys())

        assert state_keys == {
            "experts.gate_proj",
            "experts.down_proj",
            "experts.up_proj",
            "shared_expert.gate_proj",
            "shared_expert.down_proj",
            "shared_expert.up_proj",
            "router.gate.weight",
            "running_gate_stats",
            "global_gate_stats",
        }

        # Check that the state_dict can be loaded back into the model
        moe.load_state_dict(state_dict)
