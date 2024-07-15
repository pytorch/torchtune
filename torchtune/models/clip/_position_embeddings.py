# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

# TODO (@Felipe): add load hooks + interpolation on positional encodings,
# so max_num_tiles can be variable and a trained model can be adapted to a
# new value.


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images, different for every token in an image.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(self, embed_dim: int, tile_size: int, patch_size: int) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        scale = embed_dim**-0.5
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn((patch_grid_size**2 + 1, embed_dim))  # +1 for CLS token
        )

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape (..., n_tokens, embed_dim)
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """

    Token positional embedding for tiled images. There are two positional embeddings in this module:

    * local_token_positional_embedding: same for every tile, different for every token. Equivalent \
        to :class:`torchtune.models.clip._position_embeddings.TokenPositionalEmbedding`, but gated.
    * global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
    """

    def __init__(
        self, max_num_tiles: int, embed_dim: int, tile_size: int, patch_size: int
    ) -> None:
        super().__init__()
        patch_grid_size = tile_size // patch_size
        self.n_tokens_per_tile = patch_grid_size**2 + 1  # +1 for cls token
        scale = embed_dim**-0.5

        # different for every token, same for every tile
        self.local_token_positional_embedding = nn.Parameter(
            scale
            * torch.randn((patch_grid_size**2 + 1, embed_dim))  # +1 for CLS token
        )

        # different for every token, different for every tile
        self.global_token_positional_embedding = nn.Parameter(
            scale
            * torch.randn(
                max_num_tiles,
                max_num_tiles,
                self.n_tokens_per_tile,
                embed_dim,
            )
        )

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape (bsz * n_imgs, n_tiles, n_tokens, embed_dim).
            aspect_ratio (torch.Tensor): Tensor with shape (bsz * n_imgs, 2),
                where aspect_ratio[k] represents the aspect ratio of the k^th image
                of the batch before tile-cropping,  e.g. aspect_ratio[k] = (2,1).
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape

        # apply local position embedding (same for every tile)
        x = x + (self.local_token_positional_embedding * (1 - self.gate.tanh()))

        # apply global positional embedding (different for every tile)
        x = x.view(bsz_and_n_imgs, n_tiles, n_tokens, embed_dim)
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            pos_embed = self.global_token_positional_embedding[
                :n_tiles_h, :n_tiles_w, :, :
            ]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(
                n_non_padded_tiles, self.n_tokens_per_tile, embed_dim
            )
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed

        return x


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles, different for every tile, same for every token within a tile.

    Notice that tile is different from patch (token). For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each tile embedding.
    """

    def __init__(
        self,
        max_num_tiles: int,
        embed_dim: int,
    ):
        super().__init__()
        self.max_num_tiles = max_num_tiles
        self.embed_dim = embed_dim

        scale = embed_dim**-0.5
        self.embedding = nn.Parameter(
            scale * torch.randn(max_num_tiles, max_num_tiles, 1, embed_dim)
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, aspect_ratio: torch.Tensor) -> torch.Tensor:
        """
        args:
            x (torch.Tensor): Tensor with shape (bsz * n_imgs, n_tiles, n_tokens, embed_dim).
            aspect_ratio (torch.Tensor): Tensor with shape (bsz * n_imgs, 2),
                representing the aspect ratio of the image before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape

        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # When we batch images, all are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # We get only the positional encoding for non padded tiles,
            # i.e. n_tiles_h, n_tiles_w.
            pos_embed = self.embedding[:n_tiles_h, :n_tiles_w, :, :]

            # Add pos encoding to the non padded tiles.
            pos_embed = pos_embed.reshape(n_non_padded_tiles, 1, self.embed_dim)
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()

        return x
