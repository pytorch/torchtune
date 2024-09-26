# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, fixed_init_model, fixed_init_tensor
from torchtune.models.llama3_2_vision._component_builders import llama3_2_vision_decoder


@pytest.fixture
def decoder_config():
    return {
        "vocab_size": 10000,
        "num_layers": 6,
        "fusion_interval": 2,
        "num_special_tokens": 3,
        "num_heads": 8,
        "num_kv_heads": 4,
        "embed_dim": 512,
        "max_seq_len": 512,
        "encoder_max_seq_len": 512,
        "rope_base": 500000.0,
        "intermediate_dim": 2048,
    }


class TestLlama3VisionDecoder:
    @pytest.fixture(autouse=True)
    def setup_class(self, decoder_config):
        self.batch_size = 1
        self.dim = decoder_config["embed_dim"]
        self.vocab_size = decoder_config["vocab_size"]
        self.seq_len = 128
        self.input = {
            "tokens": torch.arange(self.batch_size * self.seq_len).reshape(
                self.batch_size, self.seq_len
            ),
            "encoder_input": fixed_init_tensor(
                (self.batch_size, self.seq_len, self.dim), min_val=-1, max_val=1
            ),
            "encoder_mask": None,
        }
        self.decoder = llama3_2_vision_decoder(**decoder_config)
        fixed_init_model(self.decoder, min_val=-1, max_val=1)

    @torch.no_grad()
    def test_llama3_2_vision_decoder(self):
        # call model
        output = self.decoder(**self.input)

        # assertion
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)

        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(-9.47548e-5), atol=1e-3, rtol=1e-3)
