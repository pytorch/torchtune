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
                    [0.3670, 0.2938],
                    [0.3692, 0.2921],
                    [0.3611, 0.2984],
                    [0.4207, 0.2437],
                    [0.3447, 0.3106],
                    [0.3383, 0.3150],
                    [0.3727, 0.2892],
                    [0.3996, 0.2653],
                ],
                [
                    [0.3855, 0.2783],
                    [0.2627, 0.3581],
                    [0.3601, 0.2992],
                    [0.3473, 0.3087],
                    [0.3549, 0.3032],
                    [0.2871, 0.3459],
                    [0.2753, 0.3520],
                    [0.2285, 0.3728],
                ],
            ]
        )
        assert actual.shape == (BSZ, MAX_SEQ_LEN, EMBED_DIM)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    def test_backward(self, model, inputs):
        y = model(inputs)
        loss = y.mean()
        loss.backward()
