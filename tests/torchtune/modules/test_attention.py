# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import pytest

import torch

from tests.test_utils import assert_expected, fixed_init_model
from torch import nn

from torchtune.modules import KVCache, MultiHeadAttention, RotaryPositionalEmbeddings
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestMultiHeadAttention:
    """
    Class for testing our MultiHeadAttention implementation.

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
    def input(self, input_params: Tuple[int, int, int]) -> torch.Tensor:
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
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = input_params
        _, _, _, max_seq_len = attn_params_gqa
        seq_len = max_seq_len + 1
        return torch.randn(batch_size, seq_len, embed_dim)

    @pytest.fixture
    def input_max_bs_exceeded(
        self,
        input_params: Tuple[int, int, int],
        attn_params_gqa: Tuple[int, int, int, int],
    ) -> torch.Tensor:
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
    def gqa(self, attn_params_gqa: Tuple[int, int, int, int]) -> MultiHeadAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_gqa
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
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
    ) -> MultiHeadAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_gqa
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        head_dim = embed_dim // num_heads
        kv_cache = KVCache(
            batch_size=4,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float32,
        )
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
        )
        attn.cache_enabled = True
        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mha(self, attn_params_mha: Tuple[int, int, int, int]) -> MultiHeadAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mha
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
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
    ) -> MultiHeadAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mha
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        kv_cache = KVCache(
            batch_size=4,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float32,
        )
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
        )
        attn.cache_enabled = True
        fixed_init_model(attn)
        attn.eval()
        return attn

    @pytest.fixture
    def mqa(self, attn_params_mqa: Tuple[int, int, int, int]) -> MultiHeadAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mqa
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
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
    ) -> MultiHeadAttention:
        num_heads, num_kv_heads, embed_dim, max_seq_len = attn_params_mqa
        head_dim = embed_dim // num_heads
        num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        kv_cache = KVCache(
            batch_size=4,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float32,
        )
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
        )
        attn.cache_enabled = True
        fixed_init_model(attn)
        attn.eval()
        return attn

    def test_forward_gqa(self, input: torch.Tensor, gqa: MultiHeadAttention) -> None:
        with torch.no_grad():
            output = gqa(input, input)
        assert_expected(
            output.mean(), torch.tensor(-2545.42236328125), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_gqa_kv_cache(
        self, input: torch.Tensor, gqa_kv_cache: MultiHeadAttention, attn_params_gqa
    ) -> None:

        _, _, _, max_seq_len = attn_params_gqa
        _, seq_len, _ = input.shape

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        input_pos = torch.arange(seq_len)
        mask = causal_mask[None, input_pos]

        with torch.no_grad():
            output = gqa_kv_cache(input, input, mask=mask, input_pos=input_pos)
        assert_expected(
            output.mean(), torch.tensor(-2545.42236328125), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mha(self, input: torch.Tensor, mha: MultiHeadAttention) -> None:
        with torch.no_grad():
            output = mha(input, input)
        assert_expected(
            output.mean(), torch.tensor(-2597.248046875), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mha_kv_cache(
        self, input: torch.Tensor, mha_kv_cache: MultiHeadAttention, attn_params_mha
    ) -> None:

        _, _, _, max_seq_len = attn_params_mha
        _, seq_len, _ = input.shape

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        input_pos = torch.arange(seq_len)
        mask = causal_mask[None, input_pos]

        with torch.no_grad():
            output = mha_kv_cache(input, input, mask=mask, input_pos=input_pos)
        assert_expected(
            output.mean(), torch.tensor(-2597.248046875), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mqa(self, input: torch.Tensor, mqa: MultiHeadAttention) -> None:
        with torch.no_grad():
            output = mqa(input, input)
        assert_expected(
            output.mean(), torch.tensor(-2108.07666015625), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_forward_mqa_kv_cache(
        self, input: torch.Tensor, mqa_kv_cache: MultiHeadAttention, attn_params_mqa
    ) -> None:
        _, _, _, max_seq_len = attn_params_mqa
        _, seq_len, _ = input.shape

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        input_pos = torch.arange(seq_len)
        mask = causal_mask[None, input_pos]

        with torch.no_grad():
            output = mqa_kv_cache(input, input, mask=mask, input_pos=input_pos)
        assert_expected(
            output.mean(), torch.tensor(-2108.076660156255), atol=1e-8, rtol=1e-3
        )
        assert_expected(output.shape, input.shape)

    def test_max_seq_len_exceeded(
        self,
        input_max_len_exceeded: torch.Tensor,
        gqa: MultiHeadAttention,
    ) -> None:
        with pytest.raises(Exception):
            _ = gqa(input_max_len_exceeded)

    def test_batch_size_exceeded(
        self,
        input_max_bs_exceeded: torch.Tensor,
        gqa_kv_cache: MultiHeadAttention,
    ) -> None:
        with pytest.raises(Exception):
            _ = gqa_kv_cache(input_max_bs_exceeded)
