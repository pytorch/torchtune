# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from llm.llama2.feed_forward import FeedForward

from tests.test_utils import assert_expected, set_rng_seed


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


class TestFeedForward:
    """Class for testing FFN implementation."""

    @pytest.fixture
    def input_params(self):
        dim = 4096
        hidden_dim = 4 * dim
        hidden_dim_multiple_of = 256
        ffn_dim_multiplier = None
        return dim, hidden_dim, hidden_dim_multiple_of, ffn_dim_multiplier

    @pytest.fixture
    def input(self, input_params):
        dim, _, _, _ = input_params
        return torch.randn(1, dim)

    @pytest.fixture
    def ffn(self, input_params):
        dim, hidden_dim, hidden_dim_multiple_of, ffn_dim_multiplier = input_params
        return FeedForward(dim, hidden_dim, hidden_dim_multiple_of, ffn_dim_multiplier)

    @pytest.fixture
    def ffn_with_multiplier(self, input_params):
        dim, hidden_dim, hidden_dim_multiple_of, _ = input_params
        return FeedForward(dim, hidden_dim, hidden_dim_multiple_of, 0.75)

    def test_forward(self, input, ffn):
        x_out = ffn(input)
        assert_expected(x_out.mean(), torch.tensor(0.0011))
        assert_expected(x_out.sum(), torch.tensor(4.3965))
        assert_expected(x_out.max(), torch.tensor(0.3466))

    def test_forward_with_multiplier(self, input, ffn_with_multiplier):
        x_out = ffn_with_multiplier(input)
        assert_expected(x_out.mean(), torch.tensor(0.0004))
        assert_expected(x_out.sum(), torch.tensor(1.7033))
        assert_expected(x_out.max(), torch.tensor(0.3901))
