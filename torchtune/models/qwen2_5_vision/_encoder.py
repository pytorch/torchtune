# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
from torch import nn

from torchtune.modules import Fp32LayerNorm
from torchtune.modules.transformer import _get_clones


class Qwen2_5_VisionTransformer(nn.Module):
    """
    
    """

    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        num_layers: int,
        embed_dim: int,
        layer: nn.Module,
        token_pos_embedding: nn.Module,
        pre_tile_pos_embed: Optional[nn.Module] = None,
        post_tile_pos_embed: Optional[nn.Module] = None,
        cls_projection: Optional[nn.Module] = None,
        out_indices: Optional[List[int]] = None,
        in_channels: int = 3,
        append_cls_token: bool = False,
    ) -> None:
        super().__init__()

        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if out_indices and (len(out_indices) > num_layers):
            raise ValueError(
                f"len(out_indices) must be <= num_layers. Got {out_indices=} and {num_layers=}"
            )

        # constants
        patch_grid_size = tile_size // patch_size
        self.patches_per_tile = patch_grid_size**2
        self.out_indices = out_indices
        if not out_indices:
            self.out_indices = []

        # input modules
        self.pre_tile_pos_embed = pre_tile_pos_embed
        self.post_tile_pos_embed = post_tile_pos_embed
        self.token_pos_embedding = token_pos_embedding

        self.cls_projection = cls_projection
        self.layers = _get_clones(layer, num_layers)

        # other modules
        self.conv = nn.Conv3d( #TODO: CHECK ARGS
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            bias=False,
        )

        self.ln_post = Fp32LayerNorm(embed_dim)
        self.ln_pre = Fp32LayerNorm(embed_dim)

        self.cls_token_embedding = CLSEmbedding(
            embed_dim, append_cls_token=append_cls_token
        )

    def get_image_tokens_per_tile(self):
        return self.patches_per_tile + 1  # +1 for CLS token

    def forward(
        self,
        images: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension n_imgs to the input. This is useful when a single
        sample constains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, n_imgs=2, num_tiles, n_channels, tile_size, tile_size).

        Notice that to batch it, you will have to pad n_imgs to max_n_imgs and max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, n_imgs, n_tiles, n_channels, tile_size, tile_size).
            aspect_ratio (Optional[torch.Tensor]): torch.Tensor with shape (bsz, n_imgs, 2). If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple: (x, hidden_states),
                where x is a torch.tensor of shape (bsz, n_imgs, n_tiles, n_tokens, embed_dim) and
                hidden_states has shape is a list of len(out_indices) torch.tensor with shape
                (bsz, n_imgs, n_tiles, n_tokens, embed_dim).

        Raises:
            ValueError: If aspect_ratio is None, but n_tiles > 1 in the batch.

        Examples:

            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> from torchtune.modules import VisionTransformer
            >>>
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 400
            >>> patch_size=40
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
            >>> # make it have only 1 image per sample
            >>> batch_image = tile_cropped_image.unsqueeze(1)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(1)
            >>>
            >>> # For a detailed example, please check
            >>> # torchtune.models.clip._position_embeddings.clip_vision_encoder
            >>> # model = VisionTransformer(
            ... #           out_indices = [1,2,3,4,5],
            ... #           patch_size=40,
            ... #           patch_grid_size = patch_grid_size,
            ... #           embed_dim = 32,
            ... #           num_layers = 6,
            ... #           in_channels = num_channels,
            ... #           ...)
            >>>
            >>> x, hidden_states = model(images = batch_image, aspect_ratio = batch_aspect_ratio)
            >>>
            >>> # (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(x.shape)
            torch.Size([1, 1, 2, 101, 32])
            >>>
            >>> # list with tensors of shape (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(len(hidden_states))
            5
        """
        hidden_states = []

        # parse inputs
        bsz, n_imgs, n_tiles, nch, w, h = images.shape
        bsz_and_n_imgs = bsz * n_imgs

        # if aspect_ratio is not provided, it defaults to one tile [1,1]
        if aspect_ratio is None:
            aspect_ratio = torch.ones(
                (bsz_and_n_imgs, 2), dtype=torch.int, device=images.device
            )
            if n_tiles > 1:
                raise ValueError(
                    f"aspect_ratio was not provided, but found n_tiles>1 for {images.shape=}. Please provide aspect_ratio."
                )

        images = images.reshape(bsz_and_n_imgs * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz_and_n_imgs, 2)

        # patch embeddings (tokens)
        # A tile becomes a grid of patch_grid_size X patch_grid_size patches
        # these patches are flatenned, and called tokens from here on.

        # out: (bsz * n_imgs * n_tiles, embed_dim, patch_grid_size, patch_grid_size)
        x = self.conv(images)

        # out: (bsz * n_imgs, n_tiles, n_tokens, embed_dim)
        x = x.reshape(bsz_and_n_imgs, n_tiles, -1, self.patches_per_tile).permute(
            0, 1, 3, 2
        )
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape

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
        x = x.reshape(bsz_and_n_imgs, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.layers):
            if layer_idx in self.out_indices:
                h = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
                hidden_states.append(h)
            x = transformer_layer(x)

        # norm
        x = self.ln_post(x)

        # post_tile_pos_embed
        if self.post_tile_pos_embed:
            x = x.reshape(bsz_and_n_imgs, n_tiles, n_tokens, embed_dim)
            x = self.post_tile_pos_embed(x, aspect_ratio)

        # reshape output
        x = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)

        # cls token projection. n_tokens becomes 1
        if self.cls_projection:
            x = self.cls_projection(x)

        return x, hidden_states


class CLSEmbedding(nn.Module):
    """
    Adds a CLS token to every tile in an image.

    Notice that tile is different from patch (token). An image is divided into tiles during pre-processing,
    and patches are the outcome of the convolution in the ViT applied to each tile.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
        append_cls_token (bool): If True, adds CLS token to the end of the sequence.
            Default is False, which adds CLS token to the beginning of the sequence.
    """

    def __init__(self, embed_dim: int, append_cls_token: bool = False) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.weight = nn.Parameter(scale * torch.randn(embed_dim))
        self.append_cls_token = append_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # add 1 CLS token to every tile
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        cls_emb = self.weight.broadcast_to(bsz_and_n_imgs, n_tiles, 1, embed_dim)
        return (
            torch.cat([x, cls_emb], dim=2)
            if self.append_cls_token
            else torch.cat([cls_emb, x], dim=2)
        )


class CLSProjection(nn.Module):
    """
    Linear projection of the CLS token.

    Args:
        embed_dim (int): The dimensionality of the input patch embedding.
        cls_output_dim (int): The dimensionality of the output projection.
    """

    def __init__(self, embed_dim: int, cls_output_dim: int) -> None:
        super().__init__()

        scale = embed_dim**-0.5
        self.cls_output_dim = cls_output_dim
        self.weight = nn.Parameter(scale * torch.randn(embed_dim, cls_output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_imgs, n_tiles, n_tokens, embed_dim = x.shape
        x = x.reshape(bsz * n_imgs * n_tiles, n_tokens, embed_dim)

        # out: (bsz * n_tiles, cls_output_dim)
        x = x[:, 0, :] @ self.weight

        # num_tokens becomes 1 because we only return the CLS token projection
        x = x.reshape(bsz, n_imgs, n_tiles, 1, self.cls_output_dim)
        return x
