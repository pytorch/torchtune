# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model, gpu_test
from torch import nn
from torchtune.modules.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
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
    def experts_per_token(self) -> float:
        return 2

    @pytest.fixture
    def experts(self, dim, hidden_dim, num_experts) -> int:
        return GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)

    @pytest.fixture
    def router(self, dim, num_experts, experts_per_token) -> int:
        return TokenChoiceTopKRouter(
            gate=nn.Linear(dim, num_experts, bias=False),
            dim=dim,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
        )

    @pytest.fixture
    def shared_expert(self, dim) -> int:
        return nn.Linear(dim, dim, bias=False)

    @pytest.fixture
    def moe(self, experts, router, shared_expert) -> nn.Module:
        moe = MoE(experts=experts, router=router, shared_expert=shared_expert)
        fixed_init_model(moe, min_val=-0.1, max_val=0.1)
        return moe

    @torch.no_grad()
    @gpu_test(gpu_count=1)
    def test_forward(self, moe, dim):
        """
        Test that the forward pass of the moe layer works as expected.

        [Note] has to run on GPU because torch.histc is not supported on CPU
        """
        moe.to("cuda")
        x = torch.randn((16, dim)).view(2, 8, dim).to("cuda")
        out = moe(x)

        assert out.shape == (2, 8, dim)
        assert_expected(out.mean().item(), 3303.9001, atol=1e-3, rtol=1e-3)
