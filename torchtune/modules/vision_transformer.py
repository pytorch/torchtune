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
            the size of the input image. In this case, the function will consider your image a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        embed_dim (int): The dimensionality of each patch embedding (token).
        in_channels (int): The number of image input channels.
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

        assert num_layers > 0, "num_layers must be > 0"
        assert embed_dim > 0, "embed_dim must be > 0"
        assert in_channels > 0, "in_channels must be > 0"
        assert tile_size > 0, "tile_size must be > 0"
        assert patch_size > 0, "patch_size must be > 0"
        if indices_return_hidden:
            assert (
                len(indices_return_hidden) <= num_layers
            ), f"indices_return_hidden must be <= num_layers. Got {indices_return_hidden=} and {num_layers=}"

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

        # internal modules
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

        # if aspect_ratio is not provided, it defaults to one tile [1,1]
        if aspect_ratio is None:
            aspect_ratio = torch.tensor([[1, 1]] * bsz, device=images.device)

        images = images.reshape(bsz * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz, 2)

        # patch embeddings (tokens)
        # out: (bsz, embed_dim, patch_grid_size, patch_grid_size)
        x = self.conv(images)

        # out: (bsz, patch_grid_size**2, embed_dim)
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        _, n_tokens, embed_dim = x.shape

        # pre_tile_pos_embed
        if self.pre_tile_pos_embed:
            x = x.reshape(bsz, n_tiles, n_tokens, embed_dim)
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # insert cls token
        x = x.reshape(bsz * n_tiles, n_tokens, embed_dim)
        x = self.cls_token_embedding(x)
        n_tokens += 1

        # token_pos_embedding
        x = x.reshape(bsz, n_tiles, n_tokens, embed_dim)
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
            x = x.reshape(bsz * n_tiles, n_tokens, embed_dim)
            x = self.cls_projection(x)

            # reshape to (bsz, n_tiles, num_tokens, cls_output_dim)
            # num_tokens become 1 because we only return the CLS token projection
            x = x.reshape(bsz, n_tiles, 1, -1)

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
        pseudo_bsz, _, embed_dim = x.shape  # (bsz * n_tiles, n_tokens, embed_dim)
        cls_emb = self.cls_embedding.broadcast_to(pseudo_bsz, 1, embed_dim)
        return torch.cat([cls_emb, x], dim=1)


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
        # (bsz * n_tiles, n_tokens, embed_dim) -> (bsz * n_tiles, cls_output_dim)
        return x[:, 0, :] @ self.projection
