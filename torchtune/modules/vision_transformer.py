# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from torchtune.modules import Fp32LayerNorm
from torchtune.modules.transformer import _get_clones

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

# TODO (@kartikayk): Should default values be identity instead of None?
class VisionTransformer(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, outputting of hidden layer and optional CLS projection.

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into **patches** by a convolution.
    These patches are flattened and subsequently treated as **tokens** by the transformer.

    To further enhance the performance of ViT, we support tile-cropped images, which are images divided into
    **tiles** during the preprocessing stage. For example, an 800x400 image may be cropped into two 400x400 tiles
    if the tile_size = 400. For details on preprocessing, please refer to
    ``torchtune.models.clip._transforms.CLIPImageTransform``.

    Each of these tiles is further broken down into patches by a convolution operation. For example, if
    your patch_size = 40, then each (400, 400) tile will become a grid of 10x10 patches, and your whole image will have
    num_tiles * n_tokens -> num_tiles * (10x10 patches + 1 CLS token) -> num_tiles * 101.

    Before the transformer layers, a CLS token is added to each tile as the first token.

    To help the model "see" the whole image, we use positional embeddings. If your image
    was tile-cropped, then you need to use tile positional embeddings:

    - token_pos_embedding (tiled): ``torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding``
    - pre_tile_pos_embed: ``torchtune.models.clip._position_embeddings.TilePositionalEmbedding``
    - post_tile_pos_embed: ``torchtune.models.clip._position_embeddings.TilePositionalEmbedding``

    Otherwise, pre and post tile_pos_embed should be None. For the tokens, all you need is a simple
    token positional embedding:

    - token_pos_embedding: ``torchtune.models.clip._position_embeddings.TokenPositionalEmbedding``

    All images will be considered as a stack of tiles, even if your image was not tile-cropped. In such cases,
    your image would be composed of a single tile.

    Args:
        num_layers (int): The number of transformer layers.
        layer (nn.Module): The transformer layer module.
        token_pos_embedding (nn.Module): The token positional embedding module.
        pre_tile_pos_embed (Optional[nn.Module]): The pre-tile positional embedding module. It should be
            None if your image was not tile-cropped in advance.
        post_tile_pos_embed (Optional[nn.Module]): The post-tile positional embedding module. It should be
            None if your image was not tile-cropped in advance.
        cls_projection (Optional[nn.Module]): The CLS projection module. It should take an input tensor
            of shape (bsz * n_tiles, n_tokens, embed_dim) and output a tensor of shape
            (bsz * n_tiles, cls_output_dim). If provided, only the CLS token projection will be
            outputted, instead of all tokens.
        indices_return_hidden (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, indices_return_hidden = [0, 3] will
            return the tokens before they go through the first and fourth layers.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        embed_dim (int): The dimensionality of each patch embedding (token).
        in_channels (int): The number of image input channels.

    Raises:
        ValueError: If `num_layers` is not greater than 0.
        ValueError: If `embed_dim` is not greater than 0.
        ValueError: If `in_channels` is not greater than 0.
        ValueError: If `tile_size` is not greater than 0.
        ValueError: If `patch_size` is not greater than 0.
        ValueError: If `indices_return_hidden` is not less than or equal to `num_layers`.
    """

    def __init__(
        self,
        tile_size: int,
        num_layers: int,
        layer: nn.Module,
        token_pos_embedding: nn.Module,
        pre_tile_pos_embed: Optional[nn.Module] = None,
        post_tile_pos_embed: Optional[nn.Module] = None,
        cls_projection: Optional[nn.Module] = None,
        indices_return_hidden: Optional[List[int]] = None,
        patch_size: int = 14,
        embed_dim: int = 1280,
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if indices_return_hidden and len(indices_return_hidden) > num_layers:
            raise ValueError(
                f"indices_return_hidden must be <= num_layers. Got {indices_return_hidden=} and {num_layers=}"
            )
        # constants
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.indices_return_hidden = indices_return_hidden

        # input modules
        self.pre_tile_pos_embed = pre_tile_pos_embed
        self.post_tile_pos_embed = post_tile_pos_embed
        self.token_pos_embedding = token_pos_embedding

        self.cls_projection = cls_projection
        self.transformer_layers = _get_clones(layer, num_layers)

        # other modules
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False,
        )

        self.ln_post = Fp32LayerNorm(embed_dim)
        self.ln_pre = Fp32LayerNorm(embed_dim)

        self.cls_token_embedding = CLSEmbedding(embed_dim)

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile + 1  # +1 for CLS token

    def forward(
        self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            images (torch.Tensor): If the image was tile-cropped in advance, tensor with shape
                (bsz, n_tiles, n_channels, tile_size, tile_size). Otherwise, tensor with shape
                (bsz, n_channels, tile_size, tile_size).
            aspect_ratio (Optional[torch.Tensor]): If the image was tile-cropped in advance,
                Tensor with shape (bsz, 2), representing the aspect ratio of the image
                before tile-cropping, e.g. (2,1). Otherwise, it should be None.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output tensor(s).
                If indices_return_hidden is None, it returns the token embeddings as a tensor
                of shape (bsz, n_tiles, n_tokens, embed_dim). Otherwise, it returns a
                tuple of tensors: (x, hidden_states), where hidden_states is a stack of hidden layers
                of shape (bsz, n_tiles, n_tokens, embed_dim, len(indices_return_hidden)).

        Raises:
            ValueError: If aspect_ratio is None, but n_tiles > 1 in the batch.

        Examples:

            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> from torchtune.modules import VisionTransformer
            >>>
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 400
            >>> patch_size = 40
            >>> patch_grid_size = tile_size // patch_size
            >>>
            >>> # for details about preprocessing, please check
            >>> # torchtune.models.clip._transforms.CLIPImageTransform
            >>>
            >>> # create a random image
            >>> image = torch.rand(num_channels, image_size[0], image_size[1])
            >>>
            >>> # (num_tiles, nch, h, w) -> (2, 3, 400, 400)
            >>> tile_cropped_image = tile_crop(image, tile_size)
            >>> aspect_ratio = torch.tensor([2,1])
            >>>
            >>> # make it a batch of 1 image
            >>> batch_image = tile_cropped_image.unsqueeze(0)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(0)
            >>>
            >>> # For a detailed example, please check
            >>> # torchtune.models.clip._position_embeddings.clip_vision_encoder
            >>> # model = VisionTransformer(
            ... #           indices_return_hidden = [1,2,3,4,5],
            ... #           patch_size = 40,
            ... #           patch_grid_size = patch_grid_size,
            ... #           embed_dim = 32,
            ... #           num_layers = 6,
            ... #           in_channels = num_channels,
            ... #           ...)
            >>>
            >>> x, hidden_states = model(images = batch_image, aspect_ratio = batch_aspect_ratio)
            >>>
            >>> # (bsz, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(x.shape)
            torch.Size([1, 2, 101, 32]
            >>>
            >>> # (bsz, len(indices_return_hidden), num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(hidden_states.shape)
            torch.Size([1, 5, 2, 101, 32]
        """
        hidden_states = []

        # parse inputs
        bsz, n_tiles, nch, w, h = images.shape

        # if aspect_ratio is not provided, it defaults to one tile [1,1]
        if aspect_ratio is None:
            aspect_ratio = torch.tensor([[1, 1]] * bsz, device=images.device)
            if n_tiles > 1:
                raise ValueError(
                    f"aspect_ratio was not provided, but found n_tiles>1 for {images.shape=}. Please provide aspect_ratio."
                )

        images = images.reshape(bsz * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz, 2)

        # patch embeddings (tokens)
        # A tile becomes a grid of patch_grid_size X patch_grid_size patches
        # these patches are flatenned, and called tokens from here on.

        # out: (bsz * n_tiles, embed_dim, patch_grid_size, patch_grid_size)
        x = self.conv(images)

        # out: (bsz, n_tiles, n_tokens, embed_dim)
        x = x.reshape(bsz, n_tiles, -1, self.patches_per_tile).permute(0, 1, 3, 2)
        bsz, n_tiles, n_tokens, embed_dim = x.shape

        # pre_tile_pos_embed
        if self.pre_tile_pos_embed:
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # insert cls token
        x = self.cls_token_embedding(x)
        n_tokens += 1

        # token_pos_embedding
        x = self.token_pos_embedding(x, aspect_ratio)

        # norm
        x = self.ln_pre(x)

        # transformer with optional hidden layer outputs
        x = x.reshape(bsz, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if self.indices_return_hidden and layer_idx in self.indices_return_hidden:
                hidden_states.append(x)
            x = transformer_layer(x)

        # norm
        x = self.ln_post(x)

        # reshape outputs
        x = x.reshape(bsz, n_tiles, n_tokens, embed_dim)
        if self.indices_return_hidden:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.reshape(
                bsz, n_tiles, n_tokens, embed_dim, len(self.indices_return_hidden)
            )
        else:
            hidden_states = torch.empty(0)  # dummy tensor

        # post_tile_pos_embed
        if self.post_tile_pos_embed:
            x = self.post_tile_pos_embed(x, aspect_ratio)

        # cls token projection
        if self.cls_projection:
            x = self.cls_projection(x)

        return x, hidden_states


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile in an image.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

    Args:
        embed_dim (int): The dimensionality of each patch embedding.
    Returns:
        torch.Tensor: The input tensor with added CLS tokens.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # add 1 CLS token to every tile
        bsz, n_tiles, n_tokens, embed_dim = x.shape
        cls_emb = self.cls_embedding.broadcast_to(bsz, n_tiles, 1, embed_dim)
        return torch.cat([cls_emb, x], dim=2)


class CLSProjection(nn.Module):
    """
    Linear projection of the CLS token.

    Args:
        embed_dim (int): The dimensionality of each patch embedding.
        cls_output_dim (int): The dimensionality of the output projection.
    Returns:
        torch.Tensor: The projected CLS token.
    """

    def __init__(self, embed_dim: int, cls_output_dim: int) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.projection = nn.Parameter(scale * torch.randn(embed_dim, cls_output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tiles, n_tokens, embed_dim = x.shape
        x = x.reshape(bsz * n_tiles, n_tokens, embed_dim)

        # out: (bsz * n_tiles, cls_output_dim)
        x = x[:, 0, :] @ self.projection

        # num_tokens becomes 1 because we only return the CLS token projection
        x = x.reshape(bsz, n_tiles, 1, -1)
        return x
