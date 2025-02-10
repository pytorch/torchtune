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
        expected = torch.tensor(
            [
                [
                    [0.1940, 0.5625],
                    [0.1893, 0.5681],
                    [0.2020, 0.5522],
                    [0.2547, 0.4681],
                    [0.1769, 0.5822],
                    [0.2737, 0.4281],
                    [0.2828, 0.4066],
                    [0.2841, 0.4033],
                ],
                [
                    [0.1796, 0.5792],
                    [0.2020, 0.5523],
                    [0.2209, 0.5258],
                    [0.2802, 0.4128],
                    [0.2923, 0.3817],
                    [0.2677, 0.4414],
                    [0.2458, 0.4847],
                    [0.1923, 0.5645],
                ],
            ]
        )
        assert actual.shape == (BSZ, MAX_SEQ_LEN, EMBED_DIM)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_backward(self, model, inputs):
        y = model(inputs)
        loss = y.mean()
        loss.backward()
