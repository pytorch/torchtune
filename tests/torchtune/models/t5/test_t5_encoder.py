# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtune.models.t5._component_builders import t5_encoder
from torchtune.training.seed import set_seed

VOCAB_SIZE = 512
MAX_SEQ_LEN = 8
BSZ = 2
EMBED_DIM = 2


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestT5Encoder:
    @pytest.fixture
    def model(self):
        model = t5_encoder(
            embed_dim=EMBED_DIM,
            mlp_dim=4,
            num_heads=2,
            head_dim=EMBED_DIM // 2,
            num_layers=2,
            rel_pos_num_buckets=4,
            rel_pos_max_dist=4,
            vocab_size=VOCAB_SIZE,
            norm_eps=1e-6,
            max_seq_len=MAX_SEQ_LEN,
        )

        for param in model.parameters():
            param.data.uniform_(0, 1)

        return model

    @pytest.fixture
    def inputs(self):
        return torch.randint(0, VOCAB_SIZE, (BSZ, MAX_SEQ_LEN))

    def test_forward(self, model, inputs):
        actual = model(inputs)
        print(actual)
        expected = torch.tensor(
            [
                [
                    [0.4958, 0.4845],
                    [0.4914, 0.4863],
                    [0.5089, 0.4791],
                    [0.5946, 0.4383],
                    [0.4754, 0.4925],
                    [0.6266, 0.4204],
                    [0.6327, 0.4167],
                    [0.6519, 0.4048],
                ],
                [
                    [0.4769, 0.4919],
                    [0.5096, 0.4788],
                    [0.5347, 0.4679],
                    [0.6462, 0.4085],
                    [0.6643, 0.3968],
                    [0.5970, 0.4371],
                    [0.5829, 0.4445],
                    [0.4919, 0.4861],
                ],
            ]
        )
        assert actual.shape == (BSZ, MAX_SEQ_LEN, EMBED_DIM)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_backward(self, model, inputs):
        y = model(inputs)
        loss = y.mean()
        loss.backward()
