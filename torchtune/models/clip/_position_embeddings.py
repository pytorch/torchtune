# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn


class TokenPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, patch_grid_size: int) -> None:
        super().__init__()
        scale = embed_dim**-0.5
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((patch_grid_size**2 + 1, embed_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_embedding


# torchtune/models/clip/position_embeddings.py
class GatedTokenPositionalEmbedding(nn.Module):
    def __init__(
        self, max_num_tiles: int, embed_dim: int, patch_grid_size: int
    ) -> None:
        """
        Initializes the GatedTokenPositionalEmbedding module, which is designed to handle
        positional embeddings for images divided into tiles. This module supports dynamic
        resizing of the patch_grid used for positional embeddings, in case max_num_tiles is changed.

        Notice that tiles is different than patches. An image is divided into tiles during pre-processing,
        and patches is the outcome of the convolution in the ViT applied to each tile.

        Args:
            max_num_tiles (int): The maximum number of tiles an image can be divided into.
            embed_dim (int): The dimensionality of each patch embedding.
            patch_grid_size (int): The dimensions (height, embed_dim) of the patch_grid that represents
                how many patches are in one tile.
        """

        self.num_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token

        scale = embed_dim**-0.5
        self.global_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.num_tokens_per_tile,
                embed_dim,
            )
        )

        self.local_positional_embedding = nn.Parameter(
            scale * torch.randn(patch_grid_size**2 + 1, embed_dim)
        )

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:

        # apply local position embedding (same for every tile)
        bsz, n_tiles, num_tokens, embed_dim = x.shape

        x = x.view(bsz * n_tiles, num_tokens, embed_dim)
        x = x + self.local_positional_embedding * (1 - self.gate.tanh())

        # apply global positional embedding (different for every tile)
        x = x.view(bsz, n_tiles, num_tokens, embed_dim)
        for idx, (num_tiles_height, num_tiles_width) in enumerate(aspect_ratio):
            num_non_padded_tiles = int(num_tiles_height * num_tiles_width)

            # Get positional encoding for tiles that are not padded
            _pos_embed = self.global_positional_embedding[
                :num_tiles_height, :num_tiles_width, :, :
            ]
            _pos_embed = _pos_embed.reshape(
                num_non_padded_tiles, self.num_tokens_per_tile, embed_dim
            )
            _pos_embed = _pos_embed * self.gate.tanh()
            x[idx, :num_non_padded_tiles] += _pos_embed

        return x


class TilePositionEmbedding(nn.Module):
    def __init__(
        self,
        max_num_tiles: int,
        embed_dim: int,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.embed_dim = embed_dim

        self.embedding = nn.Parameter(
            torch.randn(max_num_tiles, max_num_tiles, 1, embed_dim)
            / math.sqrt(embed_dim)
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor):
        effective_bsz = x.shape[0]  # bsz * n_tiles
        out_pos_embed = torch.zeros(
            effective_bsz, self.max_num_tiles, 1, self.embed_dim
        ).to(device=x.device, dtype=x.dtype)

        # TODO: in the other pos_emb, we sum to X inside of the for loop
        for idx, (num_tiles_height, num_tiles_width) in enumerate(aspect_ratio):
            num_non_padded_tiles = int(num_tiles_height * num_tiles_width)
            _pos_embed = self.embedding[:num_tiles_height, :num_tiles_width]
            _pos_embed = _pos_embed.reshape(num_non_padded_tiles, 1, self.width)
            out_pos_embed[idx, :num_non_padded_tiles] = _pos_embed

        x = x + out_pos_embed * self.gate.tanh()
        return x
