# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import pytest

import torch
from tests.test_utils import assert_expected, fixed_init_model
from torch import nn
from torchtune.modules.model_fusion import EarlyFusionModel, register_fusion_module
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(1)


class DummyModel(nn.Module):
    def __init__(self, dim, vocab_size):
        super().__init__()
        self.cache_enabled = False
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, vocab_size)
        register_fusion_module(self.output)

    def setup_caches(self, batch_size, dtype, *args, **kwargs):
        self.cache_enabled = True

    def caches_are_setup(self):
        return self.cache_enabled

    def reset_caches(self):
        self.cache_enabled = False

    def forward(
        self,
        tokens,
        *,
        mask=None,
        encoder_input=None,
        encoder_mask=None,
        input_pos=None,
        input_embeds=None,
    ):

        x = self.tok_embeddings(tokens) if input_embeds is None else input_embeds
        if encoder_input is not None:
            q = self.q(x)
            k = self.k(encoder_input) if encoder_input is not None else self.k(x)
            v = self.v(encoder_input) if encoder_input is not None else self.v(x)
            x += nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=encoder_mask if encoder_mask is not None else mask
            )
        x = self.output(x)
        return x


class TestEarlyFusionModel:
    @pytest.fixture
    def vocab_size(self) -> int:
        return 100

    @pytest.fixture
    def dim(self) -> int:
        return 64

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def seq_len(self) -> int:
        return 10

    @pytest.fixture
    def decoder(self, dim, vocab_size) -> nn.Module:
        decoder = DummyModel(dim, vocab_size)
        fixed_init_model(decoder, max_val=0.1)
        return decoder

    @pytest.fixture
    def fused_model(self, vocab_size, dim, decoder) -> EarlyFusionModel:
        red = nn.Embedding(vocab_size, dim)
        fixed_init_model(red)
        green = nn.Embedding(vocab_size, dim)
        fixed_init_model(green)
        blue = nn.Embedding(vocab_size, dim)
        fixed_init_model(blue)

        model = EarlyFusionModel(
            encoders={"red": red, "green": green, "blue": blue},
            decoder=decoder,
            # These are IDs that are out of vocab in the decoder
            encoder_tokens={
                "red": vocab_size,
                "green": vocab_size + 1,
                "blue": vocab_size + 2,
            },
            decoder_trainable=True,
            encoders_trainable={"red": False, "green": True, "blue": False},
            fusion_trainable=False,
        )
        return model

    @pytest.fixture
    def inputs(self, batch_size, seq_len, vocab_size):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        red_seq_len, green_seq_len, blue_seq_len = 1, 2, 3
        tokens[:, 0] = vocab_size
        tokens[:, 3:5] = vocab_size + 1
        tokens[:, 7:] = vocab_size + 2
        encoder_input = {
            "red": {"input": torch.randint(0, vocab_size, (batch_size, red_seq_len))},
            "green": {
                "input": torch.randint(0, vocab_size, (batch_size, green_seq_len))
            },
            "blue": {"input": torch.randint(0, vocab_size, (batch_size, blue_seq_len))},
        }
        encoder_mask = torch.randint(0, 2, (batch_size, seq_len, seq_len)).bool()
        input_pos = torch.Tensor([1]).int()
        return tokens, encoder_input, encoder_mask, input_pos

    @pytest.fixture
    def state_dict(self, dim, vocab_size):
        return OrderedDict(
            {
                "decoder.q.weight": torch.randn((dim, dim)),
                "decoder.q.bias": torch.randn((dim,)),
                "decoder.k.weight": torch.randn((dim, dim)),
                "decoder.k.bias": torch.randn((dim,)),
                "decoder.v.weight": torch.randn((dim, dim)),
                "decoder.v.bias": torch.randn((dim,)),
                "decoder.output.weight": torch.randn((vocab_size, dim)),
                "decoder.output.bias": torch.randn((vocab_size,)),
                "decoder.tok_embeddings.weight": torch.randn((vocab_size, dim)),
                "encoders.red.weight": torch.randn((vocab_size, dim)),
                "encoders.green.weight": torch.randn((vocab_size, dim)),
                "encoders.blue.weight": torch.randn((vocab_size, dim)),
            }
        )

    @torch.no_grad()
    def test_forward(self, fused_model, inputs, vocab_size):
        """
        Test that the forward pass of the EarlyFusionModel works as expected.
        """
        tokens, encoder_input, *_ = inputs
        batch_size, seq_len = tokens.shape

        out = fused_model(
            tokens,
            encoder_input=encoder_input,
        )

        assert out.shape == (batch_size, seq_len, vocab_size)
        assert_expected(out.mean(), torch.tensor(0.5647), atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_forward_no_decoder(self, fused_model, inputs, dim):
        """
        Test that the forward pass of the EarlyFusionModel works as expected.
        """
        tokens, encoder_input, *_ = inputs
        batch_size, seq_len = tokens.shape

        # Dummy decoder with passthrough forward and dummy tok_embeddings
        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_embeddings = (
                    lambda x: x.unsqueeze(-1).repeat(1, 1, dim).to(dtype=torch.float32)
                )

            def forward(self, **kwargs):
                return kwargs["input_embeds"]

        fused_model.decoder = DummyModule()

        out = fused_model(
            tokens,
            encoder_input=encoder_input,
        )

        assert out.shape == (batch_size, seq_len, dim)
        # Check that each encoder output is placed correctly in the fused output
        red = fused_model.encoders["red"](**encoder_input["red"])
        assert_expected(out[:, :1, :], red, atol=1e-3, rtol=1e-3)
        green = fused_model.encoders["green"](**encoder_input["green"])
        assert_expected(out[:, 3:5, :], green, atol=1e-3, rtol=1e-3)
        blue = fused_model.encoders["blue"](**encoder_input["blue"])
        assert_expected(out[:, 7:, :], blue, atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_forward_no_encoder(self, fused_model, batch_size, seq_len, vocab_size):
        """
        Test the forward pass of the EarlyFusionModel with no encoder input.
        """
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        actual = fused_model(tokens)
        expected = fused_model.decoder(
            tokens=None, input_embeds=fused_model.decoder.tok_embeddings(tokens)
        )

        assert_expected(actual, expected, atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_forward_no_decoder_uneven_encoder_tokens(
        self, fused_model, dim, batch_size, seq_len, vocab_size
    ):
        """
        If each sample has a different number of encoder tokens in the sequence, test that mask scatter
        of embeds still works as expected:

        <image> This is a dog.
        <image> My dog is better than yours. <image>
        """
        red_seq_len, green_seq_len, blue_seq_len = 1, 2, 3
        # In a real encoder input, it would be padded to max number of media in the batch, so we don't
        # make these test inputs uneven. The forward pass should still be able to take the number of embeddings
        # it needs and ignore the rest, which would be pad embeddings.
        encoder_input = {
            "red": {"input": torch.randint(0, vocab_size, (batch_size, red_seq_len))},
            "green": {
                "input": torch.randint(0, vocab_size, (batch_size, green_seq_len))
            },
            "blue": {"input": torch.randint(0, vocab_size, (batch_size, blue_seq_len))},
        }
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        # For red encoder, only the first sample has a token
        tokens[0, 0] = vocab_size
        # For green encoder, first sample has 2 tokens, second sample has 1 token
        tokens[0, 3:5] = vocab_size + 1
        tokens[1, 4] = vocab_size + 1
        # For blue encoder, first sample has 3 tokens, second sample has 2 tokens
        tokens[0, 7:] = vocab_size + 2
        tokens[1, 8:] = vocab_size + 2

        # Dummy decoder with passthrough forward and dummy tok_embeddings
        class DummyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok_embeddings = (
                    lambda x: x.unsqueeze(-1).repeat(1, 1, dim).to(dtype=torch.float32)
                )

            def forward(self, **kwargs):
                return kwargs["input_embeds"]

        fused_model.decoder = DummyModule()

        out = fused_model(
            tokens,
            encoder_input=encoder_input,
        )

        assert out.shape == (batch_size, seq_len, dim)
        # Check that each encoder output is placed correctly in the fused output
        red = fused_model.encoders["red"](**encoder_input["red"])
        assert_expected(out[0, 0, :], red[0, 0, :], atol=1e-3, rtol=1e-3)
        green = fused_model.encoders["green"](**encoder_input["green"])
        assert_expected(out[0, 3:5, :], green[0, :, :], atol=1e-3, rtol=1e-3)
        assert_expected(out[1, 4, :], green[1, 0, :], atol=1e-3, rtol=1e-3)
        blue = fused_model.encoders["blue"](**encoder_input["blue"])
        assert_expected(out[0, 7:, :], blue[0, :, :], atol=1e-3, rtol=1e-3)
        assert_expected(out[1, 8:, :], blue[1, :2, :], atol=1e-3, rtol=1e-3)

    @torch.no_grad()
    def test_decoder_forward(self, fused_model, inputs, vocab_size):
        """
        Test that the forward pass of the EarlyFusionModel works during decoding.
        """
        tokens, encoder_input, encoder_mask, input_pos = inputs
        tokens = tokens[:, input_pos]
        encoder_mask = encoder_mask[:, input_pos]
        batch_size, seq_len = tokens.shape
        out = fused_model(
            tokens,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

        assert out.shape == (batch_size, seq_len, vocab_size)
        assert_expected(out.mean(), torch.tensor(0.2383), atol=1e-3, rtol=1e-3)

    def test_setup_cache(self, fused_model):
        """
        Test that the cache methods works as expected.
        """
        fused_model.setup_caches(2, torch.float32)
        assert fused_model.caches_are_setup()
        fused_model.reset_caches()
        assert not fused_model.caches_are_setup()

    def test_set_trainable_params(self, fused_model):
        """
        Test that the trainable parameters are set correctly.
        """
        trainable_params = {
            n for n, p in fused_model.named_parameters() if p.requires_grad
        }
        assert trainable_params == {
            "decoder.q.weight",
            "decoder.q.bias",
            "decoder.k.weight",
            "decoder.k.bias",
            "decoder.v.weight",
            "decoder.v.bias",
            "decoder.tok_embeddings.weight",
            "encoders.green.weight",
        }

    def test_mismatched_encoder_tokens(self, decoder):
        with pytest.raises(ValueError):
            _ = EarlyFusionModel(
                encoders={"encoder": nn.Identity(), "encoder2": nn.Identity()},
                decoder=decoder,
                encoder_tokens={"encoder": 0, "encoder3": 1},
                encoders_trainable=False,
            )

    def test_mismatched_encoder_trainable(self, decoder):
        with pytest.raises(ValueError):
            _ = EarlyFusionModel(
                encoders={"encoder": nn.Identity(), "encoder2": nn.Identity()},
                decoder=decoder,
                encoder_tokens={"encoder": 0, "encoder2": 1},
                encoders_trainable={"encoder": True, "encoder3": False},
            )

    def test_mismatched_encoder_input(self, fused_model, inputs):
        tokens, _, _, _ = inputs
        with pytest.raises(ValueError):
            _ = fused_model(
                tokens,
                encoder_input={"encoder": {"input": torch.tensor([1])}},
            )
