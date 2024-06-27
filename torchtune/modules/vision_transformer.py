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

# torchtune/modules/vision_transformer.py
class VisionTransformer(nn.Module):
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
    ):
        super().__init__()

        # constants
        self.indices_return_hidden = indices_return_hidden
        self.patches_per_tile = patch_grid_size**2

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

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile

    def forward(
        self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        hidden_out = []

        # parse inputs
        if images.ndim == 4:
            n_tiles = 1
            bsz, nch, w, h = images.shape

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
        x = self.conv(
            images
        )  # out shape: (bsz, embed_dim, patch_grid_size, patch_grid_size)
        x = x.flatten(x, start_dim=2).permute(
            0, 2, 1
        )  # out shape: (bsz, patch_grid_size**2, embed_dim)
        _, n_patches, embed_dim = x.shape

        # pre_tile_pos_embed
        if self.pre_tile_pos_embed:
            x = x.reshape(bsz, n_tiles, n_patches, embed_dim)
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # apply cls token
        x = x.reshape(bsz * n_tiles, n_patches, embed_dim)
        x = self.cls_token_embedding(x)
        n_patches += 1

        # token_pos_embedding
        if self.token_pos_embedding:
            x = x.reshape(bsz, n_tiles, n_patches, embed_dim)
            x = self.token_pos_embedding(x)

        # norm
        x = self.ln_pre(x)

        # transformer with optional hidden layer outputs
        x = x.view(bsz, n_tiles * n_patches, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if self.indices_return_hidden and layer_idx in self.indices_return_hidden:
                hidden_out.append(x)
            x = transformer_layer(x)

        # norm
        x = self.ln_post(x)

        # reshape outputs
        x = x.reshape(bsz, n_tiles, n_patches, embed_dim)
        if self.indices_return_hidden:
            inter_x = torch.stack(hidden_out, dim=-1)
            inter_x = inter_x.reshape(bsz, n_tiles, n_patches, embed_dim)
        else:
            inter_x = torch.empty(0)

        # post_tile_pos_embed
        if self.post_tile_pos_embed:
            x = self.post_tile_pos_embed(x, aspect_ratio)

        # cls token projection
        if self.cls_projection:
            x = self.cls_projection(x)

        if self.indices_return_hidden:
            return x, inter_x
        else:
            return x


class CLSEmb(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        scale = embed_dim**-0.5
        self.cls_embedding = nn.Parameter(scale * torch.randn(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x has shape (bsz * n_tiles, n_patches, embed_dim)
        cls_emb = self.cls_embedding.to(x.dtype)

        # add 1 CLS token to every tile
        effective_bsz, embed_dim = x.shape[0], x.shape[-1]
        cls_emb = cls_emb.broadcast_to(effective_bsz, 1, embed_dim)
        return torch.cat([cls_emb, x], dim=1)


class CLSProjection(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        scale = embed_dim**-0.5
        self.projection = nn.Parameter(scale * torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0, :] @ self.projection
