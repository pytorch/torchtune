# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import fixed_init_model
from tests.torchtune.models.mistral.scripts.mistral_test_config import MistralTestConfig
from torchtune.models.mistral import mistral
from torchtune.utils.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestMistral:
    @pytest.fixture
    def inputs(self):
        return torch.randint(
            0,
            MistralTestConfig.VOCAB_SIZE,
            (MistralTestConfig.BSZ, MistralTestConfig.SEQ_LEN),
        )

    def test_forward(self, inputs):
        model = mistral(
            vocab_size=MistralTestConfig.VOCAB_SIZE,
            num_layers=MistralTestConfig.NUM_LAYERS,
            num_heads=MistralTestConfig.NUM_HEADS,
            num_kv_heads=MistralTestConfig.NUM_KV_HEADS,
            embed_dim=MistralTestConfig.EMBED_DIM,
            max_seq_len=MistralTestConfig.MAX_SEQ_LEN,
        )
        fixed_init_model(model, min_val=-0.25, max_val=0.5)
        actual = model(inputs)
        expected = torch.tensor(3.9763)
        assert actual.shape == (
            MistralTestConfig.BSZ,
            MistralTestConfig.SEQ_LEN,
            MistralTestConfig.VOCAB_SIZE,
        )
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-4)
