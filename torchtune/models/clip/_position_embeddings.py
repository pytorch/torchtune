# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

# TODO (Felipe): add load hooks + interpolation on positional encodings,
# so max_num_tiles can be variable and a trained model can be adapted to a
# new value.

"""
tiles, patches and tokens.

For our VisionTransformer, we support tile-cropping. This creates a bit of confusion
between what are tiles, patches and tokens.

Tiles are a result of the pre-processing. If you have
"""


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images. Different for every token (patch) in an image.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
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
        args:
            x (torch.Tensor): Tensor with shape (*, n_tokens, embed_dim)
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        return x + self.positional_embedding


class TiledTokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for tiled images. There are two positional embeddings in this module:
        - local_token_positional_embedding: same for every tile, different for every token. Equivalent to
            ``torchtune.models.clip._position_embeddings.TokenPositionalEmbedding``, but gated.
        - global_token_positional_embedding: different for every tile, different for every token.

    Notice that tile is different from patch. An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

    Args:
        max_num_tiles (int): The maximum number of tiles an image can be divided into.
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance.
            If your image was not tile-cropped, this embedding is not necessary.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
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
            x (torch.Tensor): Tensor with shape (bsz, n_tiles, n_tokens, embed_dim)
            aspect_ratio (torch.Tensor): Tensor with shape (bsz, 2), representing the aspect ratio of the image
                before tile-cropping, e.g. (2,1).
        Returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """
        bsz, n_tiles, n_tokens, embed_dim = x.shape

        # apply local position embedding (same for every tile)
        x = x + (self.local_token_positional_embedding * (1 - self.gate.tanh()))

        # apply global positional embedding (different for every tile)
        x = x.view(bsz, n_tiles, n_tokens, embed_dim)
        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            # For batching, all images are padded to the same amount of tiles.
            # The aspect_ratio lets us know the non padded tiles for each image.
            # We only add positional encoding to those.
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # global_token_positional_embedding is a grid of shape
            # (max_num_tiles, max_num_tiles, n_tokens, embed_dim)
            # We retrieve the positional encoding for the
            # n_tiles_h and n_tiles_w.
            # Those are the tiles that are not padded.
            pos_embed = self.global_token_positional_embedding[
                :n_tiles_h, :n_tiles_w, :, :
            ]

            # reshape pos_emebd to (n_tiles_h * n_tiles_w, n_tokens, embed_dim)
            # so we can add to the image (bsz, :n_non_padded_tiles, n_tokens, embed_dim)
            pos_embed = pos_embed.reshape(
                n_non_padded_tiles, self.n_tokens_per_tile, embed_dim
            )
            pos_embed = pos_embed * self.gate.tanh()
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed

        return x


class TilePositionalEmbedding(nn.Module):
    """
    Positional embedding for tiles. Different for every tile, same for every token within a tile.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

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
            x (torch.Tensor): Tensor with shape (bsz, n_tiles, n_tokens, embed_dim)
            aspect_ratio (torch.Tensor): Tensor with shape (bsz, 2), representing the aspect ratio of the image
                before tile-cropping, e.g. (2,1).
        returns:
            torch.Tensor: The input tensor with added positional embeddings.
        """

        for batch_idx, (n_tiles_h, n_tiles_w) in enumerate(aspect_ratio):
            n_non_padded_tiles = int(n_tiles_h * n_tiles_w)

            # get pos emb
            pos_embed = self.embedding[:n_tiles_h, :n_tiles_w, :, :]
            pos_embed = pos_embed.reshape(n_non_padded_tiles, 1, self.embed_dim)

            # update zero tensor
            x[batch_idx, :n_non_padded_tiles, :, :] += pos_embed * self.gate.tanh()

        return x
