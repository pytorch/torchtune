# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from llm.llama2.attention import LlamaSelfAttention
from torch import tensor

from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(16)


class TestLlamaSelfAttention:
    """
    Class for testing our LlamaSelfAttention implementation.

    The expected tensors are computed from the reference implementation
    below by using the same seed, same params and same initialization used
    in the fixtures below.
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450
    """

    @pytest.fixture
    def input_params(self) -> Tuple[int, int, int]:
        batch_size = 4
        seq_len = 2048
        embed_dim = 4096
        return batch_size, seq_len, embed_dim

    @pytest.fixture
    def input(self, input_params: Tuple[int, int, int]) -> tensor:
        batch_size, seq_len, embed_dim = input_params
        x = torch.randn(batch_size, seq_len, embed_dim)
        return x

    @pytest.fixture
    def attn_params_gqa(self) -> Tuple[int, int, int, int]:
        num_heads = 32
        num_kv_heads = 8
        embed_dim = 4096
        max_seq_len = 4096
        return num_heads, num_kv_heads, embed_dim, max_seq_len

    @pytest.fixture
    def attn_params_mha(self) -> Tuple[int, int, int, int]:
        num_heads = 32
        embed_dim = 4096
        max_seq_len = 4096
        return num_heads, None, embed_dim, max_seq_len

    @pytest.fixture
    def gqa(self, attn_params_gqa: Tuple[int, int, int, int]) -> LlamaSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_gqa
        attn = LlamaSelfAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        init_weights_with_constant(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mha(self, attn_params_mha: Tuple[int, int, int, int]) -> LlamaSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mha
        attn = LlamaSelfAttention(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        init_weights_with_constant(attn)
        attn.eval()
        return attn

    def test_forward_gqa(self, input: tensor, gqa: LlamaSelfAttention) -> None:
        with torch.no_grad():
            output = gqa(input)
        assert_expected(output.mean(), tensor(-10056.5293), atol=1e-8, rtol=1e-3)
        assert_expected(output.shape, input.shape)

    def test_forward_mha(self, input: tensor, mha: LlamaSelfAttention) -> None:
        with torch.no_grad():
            output = mha(input)
        assert_expected(output.mean(), tensor(-10056.5381), atol=1e-8, rtol=1e-3)
        assert_expected(output.shape, input.shape)
