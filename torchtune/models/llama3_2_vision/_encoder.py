# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from torch import nn
from torchtune.modules.model_fusion import register_fusion_module


class Llama3VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, nn.Sequential(CLIP(), Llama3VisionProjectionHead()).

    Args:
        layers (nn.Module): Transformer Decoder layers
        output (nn.Module): Output linear layer. Input dim is
            (num_hidden + 1) * encoder_dim and output is decoder_dim.
        num_hidden_inputs (int): Number of expected hidden state inputs
    """

    def __init__(
        self,
        layers: nn.Module,
        output: nn.Module,
        num_hidden_inputs: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output
        self.num_hidden = num_hidden_inputs

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x i x t x e x d]
            hidden_states (Optional[List[torch.Tensor]]): list of hidden states
                from the encoder. Each hidden state has the same shape as x.

        Returns:
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

        Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by i*t*e
            - d: embed dim
        """
        bsz, imgs, tiles, embeds, dim = x.shape

        # apply transformer layers
        x = x.view(bsz * imgs, tiles * embeds, dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(bsz, imgs, tiles, embeds, dim)

        # interleave hidden states and cat with x
        if self.num_hidden > 0:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, imgs, tiles, embeds, -1)
            x = torch.cat([x, hidden_states], dim=-1)

        # shape [b x s x d]
        x = self.output(x).reshape(bsz, imgs * tiles * embeds, -1)

        return x


class Llama3VisionEncoder(nn.Module):
    """Vision encoder model for Llama 3.2 Vision. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        clip (nn.Module): CLIP encoder vision model
        projection_head (nn.Module): projection_head that takes embeddings
            with dimension encoder_dim as input and outputs embeddings of
            size decoder_dim.
    """

    def __init__(self, clip: nn.Module, projection_head: nn.Module) -> None:
        super().__init__()
        self.clip = clip
        self.projection = projection_head
        register_fusion_module(self.projection)

    def forward(
        self, images: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b x i x t x c x w x h]
            aspect_ratio (Optional[torch.Tensor]): Tensor with shape [b x i x 2]. If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

         Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*clip_embeds_per_tile
            - d: embed dim
        """
        x, hidden_states = self.clip(images, aspect_ratio)
        x = self.projection(x, hidden_states)
        return x
