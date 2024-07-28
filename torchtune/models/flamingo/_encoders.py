# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import nn, Tensor


class FlamingoVisionAdapter(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList,
        projection: nn.Module,
    ) -> None:
        super().__init__()

        self.layers = layers
        self.projection = projection

    def forward(self, x: Tensor, hidden_states: Tensor) -> Tensor:

        bsz, n_imgs, n_tiles, n_tokens, embed_dim = x.shape

        # encoding layers
        x = x.view(bsz * n_imgs, n_tiles * n_tokens, embed_dim)
        for layers in self.layers:
            x = layers(x)

        # projection
        x = x.view(bsz, n_imgs, n_tiles, n_tokens, embed_dim)

        # stack and reshape hidden states
        num_hidden_states = len(hidden_states)
        if num_hidden_states > 0:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.reshape(
                bsz, n_imgs, n_tiles, n_tokens, embed_dim * num_hidden_states
            )
            x = torch.cat([x, hidden_states], dim=-1)

        x = self.projection(x)

        return x


class FlamingoVisionEncoder(nn.Module):
    def __init__(self, vision_encoder: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.adapter = adapter

    def forward(self, images: Tensor, aspect_ratio: Optional[Tensor] = None) -> Tensor:
        x, hidden_states = self.vision_encoder(images, aspect_ratio)
        x = self.adapter(x, hidden_states)
        return x
