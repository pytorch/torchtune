# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch
from torch import tensor

from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings
from torchtune.utils.env import seed

from tests.test_utils import assert_expected


@pytest.fixture(autouse=True)
def random():
    seed(0)


class TestRotaryPositionEmbedding:
    """
    Class for testing our Rotary Positional Embeddings (RoPE)
    implementation. The expected tensors are computed from the
    reference implementation here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450
    """

    @pytest.fixture
    def input_params(self) -> Tuple[int, int, int, int]:
        bsz = 4
        num_heads = 32
        embed_dim = 4096
        head_dim = embed_dim // num_heads
        seq_len = 2048
        max_seq_len = 4096
        return bsz, num_heads, head_dim, seq_len, max_seq_len

    @pytest.fixture
    def input(self, input_params: Tuple[int, int, int, int]) -> tensor:
        bsz, num_heads, head_dim, seq_len, _ = input_params
        return torch.randn(bsz, seq_len, num_heads, head_dim)

    @pytest.fixture
    def rope(
        self, input_params: Tuple[int, int, int, int]
    ) -> RotaryPositionalEmbeddings:
        _, _, head_dim, _, max_seq_len = input_params
        return RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

    def test_forward(self, input: tensor, rope: RotaryPositionalEmbeddings) -> None:
        x_out = rope(input)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), tensor(6.4543e-05))
        assert_expected(x_out.sum(), tensor(2165.7053))
        assert_expected(x_out.max(), tensor(5.4546))

        # check shapes
        assert_expected(x_out.shape, input.shape)

    def test_forward_with_curr_pos(
        self, input: tensor, rope: RotaryPositionalEmbeddings
    ) -> None:
        x_out = rope(input, curr_pos=10)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), tensor(0.0002), atol=1e-4)
        assert_expected(x_out.sum(), tensor(5158.3159))
        assert_expected(x_out.max(), tensor(5.4543))

        # check shapes
        assert_expected(x_out.shape, input.shape)
