# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.models.flamingo._component_builders import flamingo_decoder


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
        "rope_base": 500000.0,
        "intermediate_dim": 2048,
    }


class TestFlamingoVisionEncoder:
    def test_flamingo_text_decoder_initialization(self, decoder_config):
        # Attempt to instantiate the Flamingo text decoder
        try:
            decoder = flamingo_decoder(**decoder_config)
            print("Flamingo text decoder instantiated successfully.")
        except Exception as e:
            pytest.fail(f"Failed to instantiate Flamingo text decoder: {str(e)}")
