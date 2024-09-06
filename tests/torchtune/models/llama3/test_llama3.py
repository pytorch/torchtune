# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import fixed_init_model
from torchtune.models.llama3 import llama3
from torchtune.training.seed import set_seed

EMBED_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 16
NUM_KV_HEADS = 8
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
BSZ = 2
SEQ_LEN = 100


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLlama3:
    @pytest.fixture
    def inputs(self):
        return torch.randint(0, VOCAB_SIZE, (BSZ, SEQ_LEN))

    def test_forward(self, inputs):
        model = llama3(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            embed_dim=EMBED_DIM,
            max_seq_len=MAX_SEQ_LEN,
        )
        fixed_init_model(model, min_val=-0.25, max_val=0.5)
        actual = model(inputs)
        expected = torch.tensor(3.9763)
        assert actual.shape == (BSZ, SEQ_LEN, VOCAB_SIZE)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-4)
