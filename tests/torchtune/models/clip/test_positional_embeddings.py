# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected
from torchtune.models.clip._position_embeddings import (
    TiledTokenPositionalEmbedding,
    TilePositionalEmbedding,
    TokenPositionalEmbedding,
)

from torchtune.utils.seed import set_seed


class TestPositionalEmbeddings:
    @pytest.fixture(autouse=True)
    def setup_class(self):

        self.embed_dim = 16
        self.tile_size = 14
        self.max_num_tiles = 3
        self.batch_size = 2
        self.patch_size = 2
        self.aspect_ratio = torch.tensor([[3, 1], [1, 2]])
        self.patch_grid_size = self.tile_size // self.patch_size

        set_seed(42)
        self.input_tensor = torch.randn(
            (
                self.batch_size,
                self.max_num_tiles,
                self.patch_grid_size**2 + 1,
                self.embed_dim,
            )
        )

    def test_token_positional_embedding(self):
        # call model
        set_seed(42)
        embedding = TokenPositionalEmbedding(
            self.embed_dim, patch_size=self.patch_size, tile_size=self.tile_size
        )

        inpt = self.input_tensor.clone().reshape(
            self.batch_size * self.max_num_tiles, -1, self.embed_dim
        )
        output = embedding(inpt)

        # assertion
        assert_expected(output.shape, inpt.shape)
        assert_expected(output.mean(), torch.tensor(0.0085), atol=1e-3, rtol=1e-3)

    def test_tiled_token_positional_embedding(self):
        # call model
        set_seed(42)
        embedding = TiledTokenPositionalEmbedding(
            self.max_num_tiles,
            self.embed_dim,
            patch_size=self.patch_size,
            tile_size=self.tile_size,
        )

        # replace gate 0 -> 0.5
        embedding.gate = torch.nn.Parameter(torch.full(embedding.gate.shape, 0.5))

        inpt = self.input_tensor.clone()
        output = embedding(inpt, self.aspect_ratio)

        # assertion
        assert_expected(output.shape, self.input_tensor.shape)
        assert_expected(output.mean(), torch.tensor(0.0063), atol=1e-3, rtol=1e-3)

    def test_tile_positional_embedding(self):
        # call model
        set_seed(42)
        embedding = TilePositionalEmbedding(self.max_num_tiles, self.embed_dim)

        inpt = self.input_tensor.clone()
        output = embedding(inpt, self.aspect_ratio)

        # assertion
        assert_expected(output.shape, self.input_tensor.shape)
        assert_expected(output.mean(), torch.tensor(0.0018), atol=1e-3, rtol=1e-3)
