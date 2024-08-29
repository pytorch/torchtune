# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.modules import setup_caches
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.utils.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class DummyLayer:
    def __init__(self):
        self.cache_enabled = False

    def setup_cache(self, batch_size, dtype, encoder_max_seq_len, decoder_max_seq_len):
        self.cache_enabled = True

    def reset_cache(self):
        self.cache_enabled = False


class DummyModel(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.cache_enabled = False
        self.embed = nn.Embedding(vocab_size, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, vocab_size)
        self.max_seq_len = 2
        self.layers = [DummyLayer()]

    def caches_are_enabled(self):
        return self.layers[0].cache_enabled

    def reset_caches(self):
        for layer in self.layers:
            layer.reset_cache()

    def forward(self, tokens, mask, encoder_input, encoder_mask, input_pos):
        x = self.embed(tokens)
        if encoder_input is not None:
            q = self.q(x)
            k = self.k(encoder_input)
            v = self.v(encoder_input)
            x += nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=encoder_mask
            )
        x = self.output(x)
        return x


class TestDeepFusionModel:
    """
    Class for testing our DeepFusionModel wrapper.
    """

    @pytest.fixture
    def vocab_size(self) -> int:
        return 100

    @pytest.fixture
    def dim(self) -> int:
        return 64

    @pytest.fixture
    def encoder(self, dim, vocab_size) -> nn.Module:
        encoder = nn.Embedding(vocab_size, dim)
        fixed_init_model(encoder)
        return encoder

    @pytest.fixture
    def decoder(self, dim, vocab_size) -> nn.Module:
        decoder = DummyModel(dim, vocab_size)
        fixed_init_model(decoder, max_val=0.1)
        return decoder

    @pytest.fixture
    def fused_model(self, encoder, decoder) -> DeepFusionModel:
        model = DeepFusionModel(
            encoder=encoder,
            decoder=decoder,
        )
        return model

    @pytest.fixture
    def inputs(self, dim, vocab_size):
        batch_size = 2
        seq_len = 10
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        encoder_input = {"input": torch.randint(0, vocab_size, (batch_size, seq_len))}
        encoder_mask = torch.randint(0, 2, (batch_size, seq_len, seq_len)).bool()
        input_pos = torch.Tensor([1]).int()
        return tokens, encoder_input, encoder_mask, input_pos

    @torch.no_grad()
    def test_forward(self, fused_model, inputs, vocab_size):
        """
        Test that the forward pass of the DeepFusionModel works as expected.
        """
        tokens, encoder_input, encoder_mask, _ = inputs
        batch_size, seq_len = tokens.shape
        out = fused_model(
            tokens, encoder_input=encoder_input, encoder_mask=encoder_mask
        )

        assert out.shape == (batch_size, seq_len, vocab_size)
        assert_expected(out.mean(), torch.tensor(8.5584), atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_forward_no_encoding(self, fused_model, inputs, vocab_size):
        """
        Test that the forward pass of the DeepFusionModel with no encoder input.
        """
        tokens, *_ = inputs
        batch_size, seq_len = tokens.shape
        out = fused_model(tokens)

        assert out.shape == (batch_size, seq_len, vocab_size)
        assert_expected(out.mean(), torch.tensor(0.2271), atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_decoding_forward(self, fused_model, inputs, vocab_size):
        """
        Test that the forward pass of the DeepFusionModel works during decoding.
        """
        tokens, encoder_input, encoder_mask, input_pos = inputs
        tokens = tokens[:, input_pos]
        batch_size, seq_len = tokens.shape
        out = fused_model(
            tokens,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

        assert out.shape == (batch_size, seq_len, vocab_size)
        assert_expected(out.mean(), torch.tensor(9.0072), atol=1e-3, rtol=1e-3)

    def test_setup_cache(self, fused_model):
        """
        Test that the cache methods works as expected.
        """
        setup_caches(fused_model.decoder, batch_size=2, dtype=torch.float32)
        assert fused_model.caches_are_enabled()
        fused_model.reset_caches()
        assert not fused_model.caches_are_enabled()
