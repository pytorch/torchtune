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
        return 3

    @pytest.fixture
    def encoded(self):
        # This is 2x5x3
        encoded = tensor(
            [
                [
                    [-1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0],
                    [0.0, -1.0, -1.0],
                    [0.0, 2.0, -1.0],
                    [-2.0, -1.0, 1.0],
                ],
                [
                    [2.0, 2.0, -1.0],
                    [1.0, -1.0, -2.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 0.0, 0.0],
                ],
            ]
        )
        encoded.requires_grad_()

        return encoded

    @pytest.fixture
    def embedding_weights(self):
        # This is 4x3
        return tensor(
            [
                [1.0, 0.0, -1.0],
                [2.0, -2.0, 0.0],
                [2.0, 1.0, 0.0],
                [-1.0, -2.0, 0.0],
            ]
        )

    @pytest.fixture
    def input_tensor_flat(self):
        # This is 4x3
        return tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        )

    @pytest.fixture
    def codebook(self, num_embeddings, embedding_dim):
        return VectorQuantizedEmbeddings(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            decay=0.3,
        )

    def test_quantized_output(self, codebook, embedding_weights, encoded):
        codebook.embedding = embedding_weights
        actual = codebook(encoded)

        # This is shape (2,5,3)
        expected_quantized = tensor(
            [
                [
                    [2.0, 2.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 2.0],
                ],
                [
                    [2.0, 2.0, -1.0],
                    [1.0, -2.0, -2.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 2.0],
                    [1.0, 1.0, 0.0],
                ],
            ]
        )
        expected_token_ids = tensor([[2.0, 2.0, 0.0], [2.0, 1.0, 3.0]]).type(
            torch.LongTensor
        )

        assert_expected(actual[0], expected_quantized)
        assert_expected(actual[1], expected_token_ids)

    def test_ema_update_embedding(self, num_embeddings, embedding_dim, encoded):
        codebook = VectorQuantizedEmbeddings(
            num_embeddings, embedding_dim, learnable=True
        )
        distances = torch.cdist(encoded, codebook.embedding, p=2.0) ** 2
        codebook_indices = torch.argmin(distances, dim=1)
        codebook._ema_update_embedding(encoded, codebook_indices)

        actual_weight = codebook.embedding
        expected_weight = tensor(
            [
                [0.7647, -1.4118, 0.0000, 1.5882, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.4118, 1.4118, -0.5882, 1.1765, -1.4118],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_weight, expected_weight, rtol=0.0, atol=1e-4)

        actual_code_avg = codebook.code_avg
        expected_code_avg = tensor(
            [
                [1.3000, -2.4000, 0.0000, 2.7000, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [-0.7000, 2.4000, -1.0000, 2.0000, -2.4000],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_code_avg, expected_code_avg, rtol=0.0, atol=1e-4)

        actual_code_usage = codebook.code_usage
        expected_code_usage = tensor([1.7000, 1.0000, 1.7000, 1.0000])
        assert_expected(actual_code_usage, expected_code_usage, rtol=0.0, atol=1e-4)

    def test_codebook_restart(self, codebook, encoded):
        # Use only embedding vector at index = 1 and force restarts.
        # Slightly modify encoded_flat to make sure vectors restart to something new
        encoded_noise = encoded + torch.randn_like(encoded)
        codebook_indices_low_usage = torch.ones(encoded.shape[0], dtype=torch.long)
        codebook._ema_update_embedding(encoded_noise, codebook_indices_low_usage)

        # Check if embedding contains restarts
        for i, emb in enumerate(codebook.embedding):
            # We used only emb vector with index = 1, so check it was not restarted
            if i == 1:
                assert_expected(
                    emb,
                    codebook.code_avg[1] / codebook.code_usage[1],
                    rtol=0,
                    atol=1e-4,
                )
            # Compare each embedding vector to each encoded vector.
            # If at least one match, then restart happened.
            else:
                assert any(
                    [
                        torch.isclose(emb, enc, rtol=0, atol=1e-4).all()
                        for enc in encoded_noise
                    ]
                ), "embedding restarted from encoder output incorrectly"

    def test_lookup(self, codebook, embedding_weights):
        codebook.embedding = embedding_weights
        indices_flat = tensor([[0, 1]])  # (b, seq_len)
        indices_shaped = tensor([[[0, 1], [2, 3]]])  # (b, shape)
        actual_quantized_flat = codebook.lookup(indices_flat)
        actual_quantized = codebook.lookup(indices_shaped)
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
