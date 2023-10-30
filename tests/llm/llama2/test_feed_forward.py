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
        hidden_dim = 4096
        return dim, hidden_dim

    @pytest.fixture
    def input(self, input_params):
        dim, _ = input_params
        return torch.randn(1, dim)

    @pytest.fixture
    def ffn(self, input_params):
        dim, hidden_dim = input_params
        return FeedForward(dim, hidden_dim)

    def test_forward(self, input, ffn):
        x_out = ffn(input)
        assert_expected(x_out.mean(), torch.tensor(0.0011), atol=1e-4, rtol=1e-3)
        assert_expected(x_out.sum(), torch.tensor(4.3965), atol=1e-7, rtol=1e-3)
        assert_expected(x_out.max(), torch.tensor(0.3466), atol=1e-7, rtol=1e-3)
