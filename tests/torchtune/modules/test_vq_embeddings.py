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
    def codebook(self, num_embeddings, embedding_dim):
        def vq(learnable):
            return VectorQuantizedEmbeddings(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                decay=0.3,
                learnable=learnable,
            )

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

    def test_quantized_output(self, codebook, encoded, embedding_weights):
        vq = codebook(learnable=False)
        vq.embedding = embedding_weights
        actual = vq(encoded)

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

    def test_ema_update_embedding(self, codebook, encoded, embedding_weights):
        vq = codebook(learnable=True)
        vq.embedding = embedding_weights
        encoded_flat = encoded.view(-1, encoded.shape[-1])
        distances = torch.cdist(encoded_flat, vq.embedding, p=2.0) ** 2
        codebook_indices = torch.argmin(distances, dim=1)
        vq._ema_update_embedding(encoded_flat, codebook_indices)

        actual_weight = vq.embedding
        expected_weight = tensor(
            [
                [2.0000, -1.0000, 0.0000, 2.0000, 0.0000],
                [2.0000, 1.0000, 0.0000, 1.0000, 1.0000],
                [0.5647, 1.3760, -0.3936, 1.1213, -0.7635],
                [1.0000, 0.0000, -1.0000, -1.0000, 1.0000],
            ]
        )
        assert_expected(actual_weight, expected_weight, rtol=0.0, atol=1e-4)

        actual_code_avg = vq.code_avg
        expected_code_avg = tensor(
            [
                [0.4176, 0.3790, -0.7551, -0.6548, 1.0419],
                [1.3309, -0.3437, 0.2303, 1.1865, 0.1305],
                [1.1859, 2.8897, -0.8265, 2.3547, -1.6033],
                [-0.9834, -0.7490, -0.3521, 0.5825, 0.4301],
            ]
        )
        assert_expected(actual_code_avg, expected_code_avg, rtol=0.0, atol=1e-4)

        actual_code_usage = vq.code_usage
        expected_code_usage = tensor([0.7000, 0.7000, 2.1000, 0.7000])
        assert_expected(actual_code_usage, expected_code_usage, rtol=0.0, atol=1e-4)

    def test_codebook_restart(self, codebook, encoded, embedding_weights):
        vq = codebook(learnable=True)
        vq.embedding = embedding_weights
        # Use only embedding vector at index = 1 and force restarts.
        # Slightly modify encoded_flat to make sure vectors restart to something new
        encoded_flat = encoded.view(-1, encoded.shape[-1])
        encoded_noise = encoded_flat + torch.randn_like(encoded_flat)
        codebook_indices_low_usage = torch.ones(encoded_flat.shape[0], dtype=torch.long)
        vq._ema_update_embedding(encoded_noise, codebook_indices_low_usage)

        # Check if embedding contains restarts
        for i, emb in enumerate(vq.embedding):
            # We used only emb vector with index = 1, so check it was not restarted
            if i == 1:
                assert_expected(
                    emb,
                    vq.code_avg[1] / vq.code_usage[1],
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
        vq = codebook(learnable=False)
        vq.embedding = embedding_weights
        indices_flat = tensor([[0, 1]])  # (b, seq_len)
        indices_shaped = tensor([[[0, 1], [2, 3]]])  # (b, shape)
        actual_quantized_flat = vq.lookup(indices_flat)
        actual_quantized = vq.lookup(indices_shaped)
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
