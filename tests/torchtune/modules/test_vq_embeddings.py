# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
from tests.test_utils import assert_expected
from torch import tensor
from torchtune.modules.vq_embeddings import VectorQuantizedEmbeddings


@pytest.fixture(autouse=True)
def random_seed():
    torch.manual_seed(4)


class TestVectorQuantizedEmbeddings:
    @pytest.fixture
    def num_embeddings(self):
        return 4

    @pytest.fixture
    def embedding_dim(self):
        return 5

    @pytest.fixture
    def embedding_weights(self):
        # This is 4x5
        return tensor(
            [
                [1.0, 0.0, -1.0, -1.0, 2.0],
                [2.0, -2.0, 0.0, 0.0, 1.0],
                [2.0, 1.0, 0.0, 1.0, 1.0],
                [-1.0, -2.0, 0.0, 2.0, 0.0],
            ]
        )

    @pytest.fixture
    def codebook(self, num_embeddings, embedding_dim, embedding_weights):
        vq = VectorQuantizedEmbeddings(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        vq.embedding.data = embedding_weights
        return vq

    @pytest.fixture
    def encoded(self):
        # This is 2x3x5
        encoded = tensor(
            [
                [
                    [-1.0, 2.0, 0.0, 0.0, -2.0],
                    [0.0, 1.0, -1.0, 2.0, -1.0],
                    [1.0, 0.0, -1.0, -1.0, 1.0],
                ],
                [
                    [2.0, 1.0, 0.0, 1.0, 1.0],
                    [2.0, -1.0, 0.0, 2.0, 0.0],
                    [-1.0, -2.0, 0.0, 1.0, 0.0],
                ],
            ]
        )
        encoded.requires_grad_()

        return encoded

    def test_quantized_output(self, codebook, encoded):
        actual = codebook(encoded)

        expected_quantized = tensor(
            [
                [
                    [2.0, 1.0, 0.0, 1.0, 1.0],
                    [2.0, 1.0, 0.0, 1.0, 1.0],
                    [1.0, 0.0, -1.0, -1.0, 2.0],
                ],
                [
                    [2.0, 1.0, 0.0, 1.0, 1.0],
                    [2.0, -2.0, 0.0, 0.0, 1.0],
                    [-1.0, -2.0, 0.0, 2.0, 0.0],
                ],
            ]
        )
        expected_token_ids = tensor([[2.0, 2.0, 0.0], [2.0, 1.0, 3.0]]).type(
            torch.LongTensor
        )

        assert_expected(actual[0], expected_quantized)
        assert_expected(actual[1], expected_token_ids)

    def test_decode(self, codebook):
        indices_flat = tensor([[0, 1]])  # (b, seq_len)
        indices_shaped = tensor([[[0, 1], [2, 3]]])  # (b, shape)
        actual_quantized_flat = codebook.decode(indices_flat)
        actual_quantized = codebook.decode(indices_shaped)
        expected_quantized_flat = tensor(
            [[[1.0, 0.0, -1.0, -1.0, 2.0], [2.0, -2.0, 0.0, 0.0, 1.0]]]
        )
        expected_quantized = tensor(
            [
                [
                    [[1.0, 0.0, -1.0, -1.0, 2.0], [2.0, -2.0, 0.0, 0.0, 1.0]],
                    [[2.0, 1.0, 0.0, 1.0, 1.0], [-1.0, -2.0, 0.0, 2.0, 0.0]],
                ]
            ]
        )
        assert_expected(
            actual_quantized_flat, expected_quantized_flat, rtol=0.0, atol=1e-4
        )
        assert_expected(actual_quantized, expected_quantized, rtol=0.0, atol=1e-4)
