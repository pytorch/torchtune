# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from tests.test_utils import assert_expected

from torch import nn, Tensor

from torchtune.models.llama2 import llama2
from torchtune.models.llama2._component_builders import llama2_mlp

from torchtune.models.llama2._model_utils import scale_hidden_dim_for_mlp
from torchtune.modules import (
    CausalSelfAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from torchtune.utils.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestTransformerDecoderLayer:
    """
    Class for testing our TransformerDecoderLayer implementation.

    The expected tensors are computed from the reference implementation
    below by using the same seed, same params and same initialization used
    in the fixtures below.
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L351
    """

    @pytest.fixture
    def input_params(self) -> Tuple[int, int, int]:
        batch_size = 4
        seq_len = 2048
        embed_dim = 4096
        return batch_size, seq_len, embed_dim

    @pytest.fixture
    def input(self, input_params: Tuple[int, int, int]) -> Tensor:
        batch_size, seq_len, embed_dim = input_params
        return torch.randn(batch_size, seq_len, embed_dim)

    @pytest.fixture
    def layer_params(self) -> Tuple[int, int, int, int]:
        num_heads = 32
        num_kv_heads = 8
        embed_dim = 4096
        max_seq_len = 4096
        return num_heads, num_kv_heads, embed_dim, max_seq_len

    @pytest.fixture
    def transformer_layer(
        self, layer_params: Tuple[int, int, int, int]
    ) -> TransformerDecoderLayer:
        num_heads, num_kv_heads, embed_dim, max_seq_len = layer_params
        head_dim = embed_dim // num_heads
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
        self_attn = CausalSelfAttention(
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
        hidden_dim = scale_hidden_dim_for_mlp(embed_dim)
        mlp = llama2_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        transformer_layer = TransformerDecoderLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim),
            mlp_norm=RMSNorm(dim=embed_dim),
        )
        # TODO: fix weight initialization to use fixed_init_model
        for p in transformer_layer.parameters():
            nn.init.constant_(p, 0.05)
        transformer_layer.eval()
        return transformer_layer

    def test_forward(
        self, input: Tensor, transformer_layer: TransformerDecoderLayer
    ) -> None:
        with torch.no_grad():
            output = transformer_layer(input)
        assert_expected(output.mean(), torch.tensor(18261.0156), atol=1e-8, rtol=1e-3)
        assert_expected(output.shape, input.shape)


class TestTransformerDecoder:
    """
    Class for testing our TransformerDecoderLayer implementation.

    The expected tensors are computed from the reference implementation
    below by using the same seed, same params and same initialization used
    in the fixtures below.
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L413
    """

    @pytest.fixture
    def input_params(self) -> Tuple[int, int, int]:
        batch_size = 4
        seq_len = 512
        vocab_size = 256
        return batch_size, seq_len, vocab_size

    @pytest.fixture
    def input(self, input_params: Tuple[int, int, int]) -> Tensor:
        batch_size, seq_len, vocab_size = input_params
        return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    @pytest.fixture
    def decoder_params(self) -> Tuple[int, int, int, int, int, int]:
        vocab_size = 256
        embed_dim = 512
        num_layers = 2
        num_heads = 8
        max_seq_len = 512
        num_kv_heads = 8
        return vocab_size, embed_dim, num_layers, num_heads, max_seq_len, num_kv_heads

    @pytest.fixture
    def input_max_len_exceeded(
        self,
        input_params: Tuple[int, int, int],
        decoder_params: Tuple[int, int, int, int, int, int],
    ) -> Tensor:
        batch_size, seq_len, vocab_size = input_params
        _, _, _, _, max_seq_len, _ = decoder_params
        seq_len = max_seq_len + 1
        return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    @pytest.fixture
    def input_max_bs_exceeded(
        self,
        input_params: Tuple[int, int, int],
        decoder_params: Tuple[int, int, int, int, int, int],
    ) -> Tensor:
        batch_size, seq_len, vocab_size = input_params
        _, _, _, _, max_seq_len, _ = decoder_params
        batch_size = batch_size + 1
        return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    @pytest.fixture
    def decoder(
        self, decoder_params: Tuple[int, int, int, int, int, int]
    ) -> TransformerDecoder:
        (
            vocab_size,
            embed_dim,
            num_layers,
            num_heads,
            max_seq_len,
            num_kv_heads,
        ) = decoder_params
        decoder = llama2(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        # TODO: fix weight initialization to use fixed_init_model
        for p in decoder.parameters():
            nn.init.constant_(p, 0.2)
        decoder.eval()
        return decoder

    @pytest.fixture
    def decoder_with_kv_cache_enabled(
        self, decoder_params: Tuple[int, int, int, int, int, int]
    ) -> TransformerDecoder:
        (
            vocab_size,
            embed_dim,
            num_layers,
            num_heads,
            max_seq_len,
            num_kv_heads,
        ) = decoder_params
        decoder = llama2(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        # TODO: fix weight initialization to use fixed_init_model
        for p in decoder.parameters():
            nn.init.constant_(p, 0.2)
        decoder.eval()
        decoder.setup_caches(batch_size=4, dtype=torch.float32)
        return decoder

    def test_forward(
        self,
        input: Tensor,
        input_params: Tuple[int, int, int],
        decoder: TransformerDecoder,
    ) -> None:
        batch_size, seq_len, vocab_size = input_params
        with torch.no_grad():
            output = decoder(input)
        assert_expected(output.mean(), torch.tensor(20.4800), atol=1e-8, rtol=1e-6)
        assert_expected(output.shape, torch.Size([batch_size, seq_len, vocab_size]))

    def test_max_seq_len_exceeded(
        self,
        input_max_len_exceeded: Tensor,
        decoder: TransformerDecoder,
    ) -> None:
        with pytest.raises(Exception):
            output = decoder(input_max_len_exceeded)

    def test_kv_cache(
        self,
        input: Tensor,
        decoder_with_kv_cache_enabled: TransformerDecoder,
        decoder: TransformerDecoder,
    ) -> None:
        _, seq_len = input.shape
        input_pos = torch.arange(seq_len)

        with torch.no_grad():
            output_cache = decoder_with_kv_cache_enabled(input, input_pos=input_pos)
            output_no_cache = decoder(input)
        assert_expected(output_cache.mean(), output_no_cache.mean())

    def test_kv_cache_reset_values(
        self,
        input: Tensor,
        decoder_with_kv_cache_enabled: TransformerDecoder,
    ) -> None:
        _, seq_len = input.shape
        input_pos = torch.arange(seq_len)

        with torch.no_grad():
            _ = decoder_with_kv_cache_enabled(input, input_pos=input_pos)
            kv_cache_k_val = decoder_with_kv_cache_enabled.layers[
                0
            ].attn.kv_cache.k_cache.clone()
            kv_cache_v_val = decoder_with_kv_cache_enabled.layers[
                0
            ].attn.kv_cache.v_cache.clone()

        decoder_with_kv_cache_enabled.reset_caches()
        kv_cache_k_val_reset = decoder_with_kv_cache_enabled.layers[
            0
        ].attn.kv_cache.k_cache.clone()
        kv_cache_v_val_reset = decoder_with_kv_cache_enabled.layers[
            0
        ].attn.kv_cache.v_cache.clone()

        assert not torch.allclose(kv_cache_k_val, kv_cache_k_val_reset)
        assert not torch.allclose(kv_cache_v_val, kv_cache_v_val_reset)

    def test_kv_cache_reset_values_fails_when_not_enabled_first(
        self,
        decoder: TransformerDecoder,
    ) -> None:
        with pytest.raises(RuntimeError, match="Key value caches are not setup"):
            decoder.reset_caches()

    def test_kv_cache_batch_size_exceeded(
        self,
        input_max_bs_exceeded: Tensor,
        decoder_with_kv_cache_enabled: TransformerDecoder,
    ) -> None:
        with pytest.raises(ValueError):
            decoder_with_kv_cache_enabled(input_max_bs_exceeded)

    def test_rms_norm_propagation(
        self, decoder_params: Tuple[int, int, int, int, int, int]
    ):
        (
            vocab_size,
            embed_dim,
            num_layers,
            num_heads,
            max_seq_len,
            num_kv_heads,
        ) = decoder_params
        rms_norm_eps = 1e-2
        decoder = llama2(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            norm_eps=rms_norm_eps,
        )
        rms_norms = [m for m in decoder.modules() if isinstance(m, RMSNorm)]
        assert len(rms_norms) > 0
        for rms_norm in rms_norms:
            assert rms_norm.eps == rms_norm_eps
