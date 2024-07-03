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
"""
[tiles, patches and tokens]

We support tile-cropping in our VisionTransformer. This can be a bit hard to understand
at first glance.

Tiles: Result of a pre-processing transform that breaks the image into tiles,
so the input to a VisionTransformer can be a stack of tiles, instead of a downsized image.
E.g. if you have a 1600x1600 image, and your VisionTransformer only accepts images with shape 400x400,
instead of downsizing it to 400x400, you can tile it into 4 tiles of size 400x400. The transformer
will still be able to see the whole image if you reshape
it to (batch_size, num_tiles * num_tokens_per_tile, emb_dim).
Any image can be generalized to a one-tile image.

Patches: In the VisionTransformer, each tile is divided into patches by a convolution.
If your tile_size is 400x400, and your patch_size = 40, then each tile will become a grid of 10x10 patches.
Your total number of patches in one image will be num_tiles * num_patches_per_tile. In the case above,
4 * 100 = 400 patches.

Tokens: Each patch is flattened by the VisionTransformer. This is now called a token embedding.
A CLS token is added to each tile, and the tokens are fed to the transformers.
So each tile will have now (CLS token + num_patches_per_tile) tokens.

In summary:
1) an image is broken down into tiles during preprocessing.
2) In the ViT, the tiles will be broken down into patches.
3) The patches will be transformed. We call them tokens now, because that's how the transformer sees them.

Image: shape (8x8)
|  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
|  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
| 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
| 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
| 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 |
| 49 | 50 | 51 | 52 | 53 | 54 | 55 | 56 |
| 57 | 58 | 59 | 60 | 61 | 62 | 63 | 64 |

Tiles: shape (2,4,4) # (num_tiles, tile_size, tile_size)
|  1 |  2 |  3 |  4 |    |  5 |  6 |  7 |  8 |
|  9 | 10 | 11 | 12 |    | 13 | 14 | 15 | 16 |
| 17 | 18 | 19 | 20 |    | 21 | 22 | 23 | 24 |
| 25 | 26 | 27 | 28 |    | 29 | 30 | 31 | 32 |

| 33 | 34 | 35 | 36 |    | 37 | 38 | 39 | 40 |
| 41 | 42 | 43 | 44 |    | 45 | 46 | 47 | 48 |
| 49 | 50 | 51 | 52 |    | 53 | 54 | 55 | 56 |
| 57 | 58 | 59 | 60 |    | 61 | 62 | 63 | 64 |

Patches: shape (2,4,2,2) # (num_tiles, num_patches_per_tile, patch_size, patch_size)
|  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
|  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

| 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
| 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

| 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
| 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

| 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
| 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

token: shape (2, 4, 4) # (num_tiles, num_patches_per_tile, emb_dim)
|  1 |  2 |  9 |  10 |    |  3 |  4 |  11 |  12 |    |  5 |  6 |  13 |  14 |    |  7 |  8 |  15 |  16 |
...
...
| 49 | 50 | 57 |  58 |    | 51 |  52 | 59 |  60 |    | 53 | 54 |  61 |  62 |    | 55 | 56 |  63 |  64 |

For the positional embeddings:

TokenPositionalEmbedding and TiledTokenPositionalEmbedding.local_token_positional_embedding
Same for every tile, different for every token.
|  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
|  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
| 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
| 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

|  1 |  2 |  3 |  4 |    |  1 |  2 |  3 |  4 |
|  9 | 10 | 11 | 12 |    |  9 | 10 | 11 | 12 |
| 17 | 18 | 19 | 20 |    | 17 | 18 | 19 | 20 |
| 25 | 26 | 27 | 28 |    | 25 | 26 | 27 | 28 |

TiledTokenPositionalEmbedding.global_token_positional_embedding
Different for every tile, different for every token.
|  1 |  2 |    |  3 |  4 |    |  5 |  6 |    |  7 |  8 |
|  9 | 10 |    | 11 | 12 |    | 13 | 14 |    | 15 | 16 |

| 17 | 18 |    | 19 | 20 |    | 21 | 22 |    | 23 | 24 |
| 25 | 26 |    | 27 | 28 |    | 29 | 30 |    | 31 | 32 |

| 33 | 34 |    | 35 | 36 |    | 37 | 38 |    | 39 | 40 |
| 41 | 42 |    | 43 | 44 |    | 45 | 46 |    | 47 | 48 |

| 49 | 50 |    | 51 | 52 |    | 53 | 54 |    | 55 | 56 |
| 57 | 58 |    | 59 | 60 |    | 61 | 62 |    | 63 | 64 |

TilePositionalEmbedding
|  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
|  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
|  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |
|  1 |  1 |  1 |  1 |    |  2 |  2 |  2 |  3 |

|  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
|  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
|  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
|  3 |  3 |  3 |  3 |    |  4 |  4 |  4 |  4 |
"""


class TokenPositionalEmbedding(nn.Module):
    """
    Token positional embedding for images. DIFFERENT for every token in an image.

    Args:
        embed_dim (int): The dimensionality of each token embedding.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
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
        - local_token_positional_embedding: SAME for every tile, DIFFERENT for every token. Equivalent to
            ``torchtune.models.clip._position_embeddings.TokenPositionalEmbedding``, but gated.
        - global_token_positional_embedding: DIFFERENT for every tile, DIFFERENT for every token.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
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
    Positional embedding for tiles. DIFFERENT for every tile, SAME for every token within a tile.

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
