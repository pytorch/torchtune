# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest
import torch

from llm.llama2.position_embeddings import RotaryPositionalEmbeddings
from torch import tensor

from tests.test_utils import assert_expected, set_rng_seed


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(0)


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
        seq_len = 4096
        return bsz, num_heads, head_dim, seq_len

    @pytest.fixture
    def input(self, input_params: Tuple[int, int, int, int]) -> tensor:
        bsz, num_heads, head_dim, seq_len = input_params
        return torch.randn(bsz, seq_len, num_heads, head_dim)

    @pytest.fixture
    def rope(
        self, input_params: Tuple[int, int, int, int]
    ) -> RotaryPositionalEmbeddings:
        _, _, head_dim, seq_len = input_params
        return RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=seq_len)

    def test_forward(self, input: tensor, rope: RotaryPositionalEmbeddings) -> None:
        x_out = rope(input)

        # check the numerics of the computed tensor
        assert_expected(x_out.mean(), tensor(-4.3060e-05))
        assert_expected(x_out.sum(), tensor(-2889.6804))
        assert_expected(x_out.max(), tensor(5.6446))

        # check shapes
        assert_expected(x_out.shape, input.shape)
