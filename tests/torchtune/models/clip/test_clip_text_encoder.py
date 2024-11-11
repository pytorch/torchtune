# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtune.models.clip._text_encoder import CLIPTextEncoder
from torchtune.training.seed import set_seed

VOCAB_SIZE = 512
MAX_SEQ_LEN = 77
BSZ = 2
EMBED_DIM = 4


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestClipTextEncoder:
    @pytest.fixture
    def model(self):
        model = CLIPTextEncoder(
            vocab_size=VOCAB_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            embed_dim=EMBED_DIM,
            num_heads=2,
            num_layers=2,
        )

        for param in model.parameters():
            param.data.uniform_(0, 1)

        return model

    @pytest.fixture
    def inputs(self):
        return torch.randint(0, VOCAB_SIZE, (BSZ, MAX_SEQ_LEN))

    def test_forward(self, model, inputs):
        actual = model(inputs)
        expected = torch.tensor(
            [[0.1850, 0.8112, 1.4308, 0.0021], [0.1903, 0.8700, 1.3892, -0.6564]]
        )
        assert actual.shape == (BSZ, EMBED_DIM)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_backward(self, model, inputs):
        y = model(inputs)
        loss = y.mean()
        loss.backward()
