# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch

from torch import nn, Tensor
from torchtune.modules.model_fusion import register_fusion_module
from torchtune.modules.transformer import _get_clones


class FlamingoProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained encoder to a pretrained decoder model.

    Args:
        layer (nn.Module): Transformer Decoder layer
        num_layers (int): Number of Transformer Decoder layers
        output (nn.Module): Output linear layer. Input dim is
            (num_hidden + 1) * encoder_dim and output is decoder_dim.
        num_hidden_inputs (int): Number of expected hidden state inputs
    """

    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        output: nn.Module,
        num_hidden_inputs: int = 0,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(layer, num_layers)
        self.output = output
        self.num_hidden = num_hidden_inputs

    def forward(
        self, x: Tensor, hidden_states: Optional[List[Tensor]] = None
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x num_imgs x num_tiles x num_embeds x encoder_dim]
            hidden_states (Optional[List[Tensor]]): list of hidden states
                from the encoder. Each hidden state has the same shape as x.

        Returns:
            Tensor: output tensor of a sequence of embedings
                [batch_size x encoder_seq_length x decoder_dim]
                where sequence length is num_imgs*num_tiles+num_embeds
        """
        bsz, imgs, tiles, embeds, dim = x.shape

        # apply transformer layers
        x = x.view(bsz * imgs, tiles * embeds, dim)
        for layers in self.layers:
            x = layers(x)
        x = x.view(bsz, imgs, tiles, embeds, dim)

        # concat hidden states
        if self.num_hidden > 0:
            x = torch.cat([x, *hidden_states], dim=-1)

        # shape [batch_size x encoder_seq_length x decoder_dim]
        x = self.output(x).reshape(bsz, imgs * tiles * embeds, -1)

        return x


class FlamingoEncoder(nn.Module):
    """Vision encoder model for Flamingo. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        encoder (nn.Module): encoder vision model
        projection_head (nn.Module): projection_head that takes embeddings
            with dimension encoder_dim as input and outputs embeddings of
            size decoder_dim.
    """

    def __init__(self, encoder: nn.Module, projection_head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = projection_head
        register_fusion_module(self.projection)

    def forward(self, images: Tensor, aspect_ratio: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            images (Tensor): Image tensor with shape
                [batch_size x num_imgs x num_tiles x num_channels x tile_size x tile_size]
            aspect_ratio (Optional[Tensor]): Tensor with shape [batch_size x num_imgs x 2]. If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tensor: output tensor of a sequence of embedings
                [batch_size x encoder_seq_length x decoder_dim]
                where sequence length is num_imgs*num_tiles+num_embeds
        """
        x, hidden_states = self.encoder(images, aspect_ratio)
        x = self.projection(x, hidden_states)
        return x
