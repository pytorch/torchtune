# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch
from torch import Tensor

from torchtune.modules import FeedForward
from torchtune.utils.env import seed

from tests.test_utils import assert_expected, fixed_init_model


@pytest.fixture(autouse=True)
def random():
    seed(0)


class TestFeedForward:
    """Class for testing FFN implementation."""

    @pytest.fixture
    def input_params(self) -> Tuple[int, int]:
        dim = 4096
        hidden_dim = 11008  # Scaled for SwiGLU
        return dim, hidden_dim

    @pytest.fixture
    def input(self, input_params: Tuple[int, int]) -> Tensor:
        dim, _ = input_params
        return torch.randn(1, dim)

    @pytest.fixture
    def ffn(self, input_params: Tuple[int, int]) -> FeedForward:
        dim, hidden_dim = input_params
        ff = FeedForward(dim, hidden_dim, linear_class=torch.nn.Linear).eval()
        fixed_init_model(ff)
        ff.eval()
        return ff

    def test_forward(self, input: Tensor, ffn: FeedForward) -> None:
        with torch.no_grad():
            x_out = ffn(input)
        assert_expected(x_out.mean(), torch.tensor(251.5356), atol=1e-7, rtol=1e-3)
        assert_expected(x_out.max(), torch.tensor(503.0614), atol=1e-7, rtol=1e-3)
