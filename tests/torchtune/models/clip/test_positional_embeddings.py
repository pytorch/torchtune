# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected, fixed_init_model, fixed_init_tensor
from torchtune.models.clip._position_embeddings import (
    TiledTokenPositionalEmbedding,
    TilePositionalEmbedding,
    TokenPositionalEmbedding,
)


class TestPositionalEmbeddings:
    @pytest.fixture(autouse=True)
    def setup_class(self):

        self.embed_dim = 16
        self.tile_size = 14
        self.max_num_tiles = 3
        self.bsz_and_n_imgs = 2
        self.patch_size = 2
        self.aspect_ratio = torch.tensor([[3, 1], [1, 2]])
        self.patch_grid_size = self.tile_size // self.patch_size

        input_tensor = torch.randn(
            (
                self.bsz_and_n_imgs,
                self.max_num_tiles,
                self.patch_grid_size**2 + 1,
                self.embed_dim,
            )
        )
        self.input_tensor = fixed_init_tensor(input_tensor.shape, min_val=-1, max_val=1)

    def test_token_positional_embedding(self):
        # call model
        embedding = TokenPositionalEmbedding(
            self.embed_dim, patch_size=self.patch_size, tile_size=self.tile_size
        )
        fixed_init_model(embedding, min_val=-1, max_val=1)

        inpt = self.input_tensor.clone().reshape(
            self.bsz_and_n_imgs * self.max_num_tiles, -1, self.embed_dim
        )
        output = embedding(inpt)

        # assertion
        assert_expected(output.shape, inpt.shape)
        assert_expected(output.mean(), torch.tensor(-0.001458), atol=1e-3, rtol=1e-3)

    def test_tiled_token_positional_embedding(self):
        # call model
        embedding = TiledTokenPositionalEmbedding(
            self.max_num_tiles,
            self.embed_dim,
            patch_size=self.patch_size,
            tile_size=self.tile_size,
        )
        fixed_init_model(embedding, min_val=-1, max_val=1)

        # replace gate 0 -> 0.5
        embedding.gate = torch.nn.Parameter(torch.full(embedding.gate.shape, 0.5))

        inpt = self.input_tensor.clone()
        output = embedding(inpt, self.aspect_ratio)

        # assertion
        assert_expected(output.shape, self.input_tensor.shape)
        assert_expected(output.mean(), torch.tensor(-0.17208), atol=1e-3, rtol=1e-3)

    def test_tile_positional_embedding(self):
        # call model
        embedding = TilePositionalEmbedding(self.max_num_tiles, self.embed_dim)
        fixed_init_model(embedding, min_val=-1, max_val=1)

        inpt = self.input_tensor.clone()
        output = embedding(inpt, self.aspect_ratio)

        # assertion
        assert_expected(output.shape, self.input_tensor.shape)
        assert_expected(output.mean(), torch.tensor(0.28627), atol=1e-3, rtol=1e-3)
