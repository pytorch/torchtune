# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import pytest

import torch

from llm.llama2.transformer import TransformerDecoder, TransformerDecoderLayer

from torch import Tensor

from tests.test_utils import assert_expected, init_weights_with_constant, set_rng_seed


@pytest.fixture(autouse=True)
def random():
    set_rng_seed(16)


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
        transformer_layer = TransformerDecoderLayer(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        init_weights_with_constant(transformer_layer, constant=0.05)
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
    def input_params(self) -> Tuple[int, int]:
        batch_size = 4
        seq_len = 2048
        vocab_size = 1024
        return batch_size, seq_len, vocab_size

    @pytest.fixture
    def input(self, input_params: Tuple[int, int]) -> Tensor:
        batch_size, seq_len, vocab_size = input_params
        return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))

    @pytest.fixture
    def decoder_params(self) -> Tuple[int, int, int, int, int, int]:
        vocab_size = 1024
        embed_dim = 4096
        num_layers = 4
        num_heads = 32
        max_seq_len = 4096
        num_kv_heads = 8
        return vocab_size, embed_dim, num_layers, num_heads, max_seq_len, num_kv_heads

    @pytest.fixture
    def input_max_len_exceeded(
        self,
        input_params: Tuple[int, int],
        decoder_params: Tuple[int, int, int, int, int, int],
    ) -> Tensor:
        batch_size, seq_len, vocab_size = input_params
        _, _, _, _, max_seq_len, _ = decoder_params
        seq_len = max_seq_len + 1
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
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        init_weights_with_constant(decoder, constant=0.2)
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
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            max_bsz_for_kv_cache=32,
        )
        init_weights_with_constant(decoder, constant=0.2)
        decoder.eval()
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
        assert_expected(output.mean(), torch.tensor(163.8399), atol=1e-8, rtol=1e-6)
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
        input_params: Tuple[int, int, int],
        decoder_with_kv_cache_enabled: TransformerDecoder,
        decoder: TransformerDecoder,
    ) -> None:
        with torch.no_grad():
            output_cache = decoder_with_kv_cache_enabled(input)
            output_no_cache = decoder(input)
        assert_expected(output_cache.mean(), output_no_cache.mean())
