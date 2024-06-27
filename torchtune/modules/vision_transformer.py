# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torchtune.modules import LayerNorm
from torchtune.modules.transformer import _get_clones

logger = logging.getLogger(__name__)


class VisionTransformer(nn.Module):
    """
    Implementation of the ViT architecture (https://arxiv.org/abs/2010.11929),
    with support for tile-cropped images, optional hidden layer outputs and optional CLS projection.

    If your images input are not tile-cropped, they are considered them as single-time images.

    ViT is a transformer architecture that takes in images and outputs N embedded tokens that
    represent this image. Each image is divided into patches by a convolution.
    These patches are flattened and then treated as tokens by a transformer.

    To further enhance the performance of ViT, we support tile-cropped images, which are images divided into
    tiles during the preprocessing stage. For example, an 800x400 image is divided
    into 4x2 tiles of 200x200, if your tile_size = 200 (for preprocessing details,
    ``check torchtune.models.clip._transforms.CLIPImageTransform``).

    Each of these tiles are treated as a single image, and further broken down into patches. For example, if
    your patch_size = 20, then each tile will have 10x10 patches, and your whole image will have
    4x2x20 patches (tokens).

    To make help the model "see" the whole image, we have 3 types of positional embeddings:
        - pre_tile_pos_embed (``torchtune.models.clip._position_embeddings.TokenPositionalEmbedding``)
        - post_tile_pos_embed (``torchtune.models.clip._position_embeddings.TokenPositionalEmbedding``)
        - token_pos_embedding (``torchtune.models.clip._position_embeddings.TiledTokenPositionalEmbedding``)

    If your images are not tile-cropped, then you do NOT need tile positional embeddings.
    tile_pos_embed should be None and token_pos_embedding should
    be ``torchtune.models.clip._position_embeddings.TokenPositionalEmbedding``.

    Args:
        patch_grid_size (int): The size of a squared grid that represents how many
             patches are in one tile, i.e. tile_size // patch_size.
        num_layers (int): The number of transformer layers.
        layer (nn.Module): The transformer layer module.
        token_pos_embedding (nn.Module): The token positional embedding module.
        pre_tile_pos_embed (Optional[nn.Module]): The pre-tile positional embedding module. It should be
            None if your image was not tile-cropped.
        post_tile_pos_embed (Optional[nn.Module]): The post-tile positional embedding module. It should be
            None if your image was not tile-cropped.
        cls_projection (Optional[nn.Module]): The CLS projection module. It should take an input tensor
            of shape (bsz, n_tokens, embed_dim) and output a tensor of shape (bsz, embed_dim).
        indices_return_hidden (Optional[List[int]]): The indices of hidden layers to return. These
            hidden layers are not part of the cls_projection. Notice that it returns the indice
            BEFORE it goes through the transformer layer.
        patch_size (int): The size of each square patch. E.g. for patch_size = 20, a tile of size 100x100
            will have 5x5 patches.
        embed_dim (int): The dimensionality of each patch embedding (token).
        in_channels (int): The number of input channels.
    """

    def __init__(
        self,
        patch_grid_size: int,
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

        # constants
        self.patches_per_tile = patch_grid_size**2
        self.indices_return_hidden = indices_return_hidden

        # input modules
        self.pre_tile_pos_embed = pre_tile_pos_embed
        self.post_tile_pos_embed = post_tile_pos_embed
        self.token_pos_embedding = token_pos_embedding

        self.cls_projection = cls_projection
        self.transformer_layers = _get_clones(layer, num_layers)

        # built-in modules
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=False,
        )

        self.ln_post = LayerNorm(embed_dim)
        self.ln_pre = LayerNorm(embed_dim)

        self.cls_token_embedding = CLSEmbedding(embed_dim)

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile

    def forward(
        self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        Args:
            images (torch.Tensor): Tensor with shape (bsz, n_tiles, n_channels, w, h), if the image was
                tile-cropped, or (bsz, n_channels, w, h) otherwise.
            aspect_ratio (Optional[torch.Tensor]): Tensor with shape (bsz, 2). It should be None or 1x1
                if the image was NOT tile-cropped.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: The output tensor(s).
                If indices_return_hidden is None, then it returns a single tensor
                of shape (bsz, n_tiles, n_tokens, embed_dim). Otherwise, it returns a
                tuple of tensors: (x, hidden_out), where hidden_out is a stack of hidden layers,
                also of shape (bsz, n_tiles, n_tokens, embed_dim). If your image is not tiled_cropped,
                n_tiles will be 1.
        Examples:
            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 200
            >>>
            >>> # for details on preprocessing, check torchtune.models.clip._transforms.CLIPImageTransform
            >>> image = torch.rand(num_channels, image_size[0], image_size[1])
            >>> tile-cropped_image = tile_crop(image, tile_size) # 4x2 tiles -> (8, 2, 200, 200)
            >>> aspect_ratio = torch.tensor([4,2])
            >>>
            >>> batch_image = tile-cropped_image.unsqueeze(0) # (1, 8, 2, 200, 200)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(0) # (1, 2)
            >>>
            >>> # model = VisionTransformer(
            ...            indices_return_hidden = [3,6,9],
            ...            patch_size = 10,
            ...            patch_grid_size = 10,
            ...            embed_dim = 1280,
            ...            ...)
            >>> x, hidden_out = model(images = batch_image, aspect_ratio = batch_aspect_ratio)
            >>> x.shape, hidden_out.shape
            ((1, 8, 100, 1280), (1, 4, 8, 100, 1280))

        """
        hidden_out = []

        # parse inputs
        if images.ndim == 4:
            n_tiles = 1
            bsz, nch, w, h = images.shape
            assert (
                aspect_ratio is None or aspect_ratio.max() == 1
            ), f"since images.ndim == 4, aspect ratio must be none or 1x1, got {aspect_ratio}"

        elif images.ndim == 5:
            bsz, n_tiles, nch, w, h = images.shape

        else:
            raise ValueError(
                f"Unsupported number of dimensions: {images.ndim}. Expected 4 or 5."
            )

        # TODO: do i need to verify that w == h?
        # if aspect_ratio is not provided, it defaults to one tile [1,1]
        if aspect_ratio is None:
            aspect_ratio = torch.tensor([[1, 1]] * bsz, device=images.device)

        images = images.reshape(bsz * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz, 2)

        # patch embeddings (tokens)

        # out shape: (bsz, embed_dim, patch_grid_size, patch_grid_size)
        x = self.conv(images)

        # out shape: (bsz, patch_grid_size**2, embed_dim)
        x = x.flatten(x, start_dim=2).permute(0, 2, 1)

        _, n_tokens, embed_dim = x.shape

        # pre_tile_pos_embed
        if self.pre_tile_pos_embed:
            x = x.reshape(bsz, n_tiles, n_tokens, embed_dim)
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # apply cls token
        x = x.reshape(bsz * n_tiles, n_tokens, embed_dim)
        x = self.cls_token_embedding(x)
        n_tokens += 1

        # token_pos_embedding
        if self.token_pos_embedding:
            x = x.reshape(bsz, n_tiles, n_tokens, embed_dim)
            x = self.token_pos_embedding(x)

        # norm
        x = self.ln_pre(x)

        # transformer with optional hidden layer outputs
        x = x.view(bsz, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if self.indices_return_hidden and layer_idx in self.indices_return_hidden:
                hidden_out.append(x)
            x = transformer_layer(x)

        # norm
        x = self.ln_post(x)

        # reshape outputs
        x = x.reshape(bsz, n_tiles, n_tokens, embed_dim)

        if self.indices_return_hidden:
            hidden_out = torch.stack(hidden_out, dim=-1)
            hidden_out = hidden_out.reshape(bsz, n_tiles, n_tokens, embed_dim)
        else:
            hidden_out = torch.empty(0)  # dummy tensor

        # post_tile_pos_embed
        if self.post_tile_pos_embed:
            x = self.post_tile_pos_embed(x, aspect_ratio)

        # cls token projection
        if self.cls_projection:
            x = x.reshape(bsz * n_tiles, n_tokens, embed_dim)
            x = self.cls_projection(x)
            x = x.reshape(bsz, n_tiles, 1, embed_dim)

        if self.indices_return_hidden:
            return x, hidden_out
        else:
            return x


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

        cls_emb = self.cls_embedding.to(x.dtype)

        # add 1 CLS token to every tile
        effective_bsz, _, embed_dim = x.shape  # (bsz * n_tiles, n_tokens, embed_dim)
        cls_emb = cls_emb.broadcast_to(effective_bsz, 1, embed_dim)
        return torch.cat([cls_emb, x], dim=1)


class CLSProjection(nn.Module):
    """
    Linear projection of the CLS token.

    Args:
        embed_dim (int): The dimensionality of each patch embedding.
    Returns:
        torch.Tensor: The projected CLS token.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.projection = nn.Parameter(scale * torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (bsz * n_tiles, n_tokens, embed_dim) -> (bsz * n_tiles, embed_dim)
        return x[:, 0, :] @ self.projection
