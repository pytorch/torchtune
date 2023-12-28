# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import pytest

import torch
from torch import nn, Tensor

from torchtune.modules import CausalSelfAttention, KVCache, RotaryPositionalEmbeddings
from torchtune.utils.env import seed

from tests.test_utils import assert_expected, fixed_init_model


@pytest.fixture(autouse=True)
def random():
    seed(16)


class TestCausalSelfAttention:
    """
    Class for testing our CausalSelfAttention implementation.

    The expected tensors are computed from the reference implementation
    below by using the same seed, same params and same initialization used
    in the fixtures below.
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450
    """

    def _get_mask(self, inpt: Tensor) -> Tensor:
        seq_len = inpt.shape[1]
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=inpt.device)
        mask = torch.triu(mask, diagonal=1).type_as(inpt)
        return mask

    @pytest.fixture
    def input_params(self) -> Tuple[int, int, int]:
        batch_size = 4
        seq_len = 2048
        embed_dim = 4096
        return batch_size, seq_len, embed_dim

    @pytest.fixture
    def input(self, input_params: Tuple[int, int, int]) -> Tensor:
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
    def input_max_len_exceeded(
        self,
        input_params: Tuple[int, int, int],
        attn_params_gqa: Tuple[int, int, int, int],
    ) -> Tensor:
        batch_size, seq_len, embed_dim = input_params
        _, _, _, max_seq_len = attn_params_gqa
        seq_len = max_seq_len + 1
        return torch.randn(batch_size, seq_len, embed_dim)

    @pytest.fixture
    def input_max_bs_exceeded(
        self,
        input_params: Tuple[int, int, int],
        attn_params_gqa: Tuple[int, int, int, int],
    ) -> Tensor:
        batch_size, seq_len, embed_dim = input_params
        _, _, _, max_seq_len = attn_params_gqa
        batch_size += 1
        return torch.randn(batch_size, seq_len, embed_dim)

    @pytest.fixture
    def attn_params_mha(self) -> Tuple[int, Optional[int], int, int]:
        num_heads = 32
        embed_dim = 4096
        max_seq_len = 4096
        return num_heads, None, embed_dim, max_seq_len

    @pytest.fixture
    def attn_params_mqa(self) -> Tuple[int, int, int, int]:
        num_heads = 32
        num_kv_heads = 1
        embed_dim = 4096
        max_seq_len = 4096
        return num_heads, num_kv_heads, embed_dim, max_seq_len

    @pytest.fixture
    def gqa(self, attn_params_gqa: Tuple[int, int, int, int]) -> CausalSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_gqa
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
        )

        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def gqa_kv_cache(
        self, attn_params_gqa: Tuple[int, int, int, int]
    ) -> CausalSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_gqa
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        head_dim = embed_dim // num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        kv_cache = KVCache(
            max_batch_size=4,
            max_seq_len=max_seq_len,
            n_kv_heads=num_heads,
            head_dim=head_dim,
        )
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
        )
        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mha(self, attn_params_mha: Tuple[int, int, int, int]) -> CausalSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mha
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
        )
        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mha_kv_cache(
        self, attn_params_mha: Tuple[int, int, int, int]
    ) -> CausalSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mha
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        kv_cache = KVCache(
            max_batch_size=4,
            max_seq_len=max_seq_len,
            n_kv_heads=num_heads,
            head_dim=head_dim,
        )
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
        )
        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mqa(self, attn_params_mqa: Tuple[int, int, int, int]) -> CausalSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mqa
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
        )
        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mqa_kv_cache(
        self, attn_params_mqa: Tuple[int, int, int, int]
    ) -> CausalSelfAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mqa
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        kv_cache = KVCache(
            max_batch_size=4,
            max_seq_len=max_seq_len,
            n_kv_heads=num_heads,
            head_dim=head_dim,
        )
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
        )
        fixed_init_model(attn)
        attn.eval()
        return attn

    def test_forward_gqa(self, input: Tensor, gqa: CausalSelfAttention) -> None:
        with torch.no_grad():
            output = gqa(input)
        assert_expected(
            output.mean(), torch.tensor(-2852.556640625), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_gqa_kv_cache(
        self, input: Tensor, gqa_kv_cache: CausalSelfAttention
    ) -> None:
        # seq_len = input.shape[1]
        # mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=input.device)
        # mask = torch.triu(mask, diagonal=1).type_as(input)
        mask = self._get_mask(input)
        with torch.no_grad():
            output = gqa_kv_cache(input, mask=mask, curr_pos=0)
        assert_expected(
            output.mean(), torch.tensor(-2852.556640625), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mha(self, input: Tensor, mha: CausalSelfAttention) -> None:
        with torch.no_grad():
            output = mha(input)
        assert_expected(
            output.mean(), torch.tensor(-2598.19482421875), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mha_kv_cache(
        self, input: Tensor, mha_kv_cache: CausalSelfAttention
    ) -> None:
        mask = self._get_mask(input)
        with torch.no_grad():
            output = mha_kv_cache(input, mask=mask, curr_pos=0)
        assert_expected(
            output.mean(), torch.tensor(-2598.19482421875), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mqa(self, input: Tensor, mqa: CausalSelfAttention) -> None:
        with torch.no_grad():
            output = mqa(input)
        assert_expected(
            output.mean(), torch.tensor(-5087.19775390625), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mqa_kv_cache(
        self, input: Tensor, mqa_kv_cache: CausalSelfAttention
    ) -> None:
        mask = self._get_mask(input)
        with torch.no_grad():
            output = mqa_kv_cache(input, mask=mask, curr_pos=0)
        assert_expected(
            output.mean(), torch.tensor(-5087.19775390625), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_max_seq_len_exceeded(
        self,
        input_max_len_exceeded: Tensor,
        gqa: CausalSelfAttention,
    ) -> None:
        with pytest.raises(Exception):
            output = gqa(input_max_len_exceeded)

    def test_max_batch_size_exceeded(
        self,
        input_max_bs_exceeded: Tensor,
        gqa_kv_cache: CausalSelfAttention,
    ) -> None:
        with pytest.raises(Exception):
            _ = gqa_kv_cache(input_max_bs_exceeded)
