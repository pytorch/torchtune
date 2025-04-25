# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from torch import nn
from torchtune.modules.model_fusion import register_fusion_module


class Llama4VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, ``nn.Sequential(CLIP(), Llama4VisionProjectionHead())``.

    Note: this module assumes the CLS token embedding is added at the end
    of the sequence.

    Args:
        output (nn.Module): output layer, typically an MLP.
        pixel_shuffle_scaling_factor (float): scaling factor for pixel shuffle.
    """

    def __init__(
        self,
        output: nn.Module,
        pixel_shuffle_scaling_factor: float = 0.5,
    ) -> None:
        super().__init__()
        self.output = output
        self.pixel_shuffle_scaling_factor = pixel_shuffle_scaling_factor

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(
            n,
            w,
            int(h * self.pixel_shuffle_scaling_factor),
            int(c / self.pixel_shuffle_scaling_factor),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * self.pixel_shuffle_scaling_factor),
            int(w * self.pixel_shuffle_scaling_factor),
            int(
                c
                / (
                    self.pixel_shuffle_scaling_factor
                    * self.pixel_shuffle_scaling_factor
                )
            ),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b, e, d]

        Returns:
            Tensor: output tensor of a sequence of embeddings [b, s, d * pixel_shuffle_factor ** 2]

        Notation used for tensor shapes:
            - b: batch size
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by t * (e - 1) // (pixel_shuffle_factor ** 2)
            - d: embed dim
        """
        # Remove cls token - assumes it is the last token in the sequence
        x = x[:, :-1, :]
        bsz, embeds, dim = x.shape

        # apply pixel shuffle
        h_patches = w_patches = int(embeds**0.5)
        x = x.reshape(bsz, h_patches, w_patches, -1)
        x = self._pixel_shuffle(x)
        _, new_h_patches, new_w_patches, new_dim = x.shape
        # shape: [bsz, embeds // factor ** 2, dim * factor ** 2)]
        x = x.reshape(bsz, new_h_patches * new_w_patches, new_dim)
        # apply output - shape [bsz, embeds // factor ** 2, output_dim]
        x = self.output(x)

        return x


class Llama4VisionEncoder(nn.Module):
    """Vision encoder model for Llama 4. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        clip (nn.Module): CLIP encoder vision model
        projection_head (nn.Module): ``projection_head`` that takes embeddings
            with dimension ``encoder_dim`` as input and outputs embeddings of
            size ``decoder_dim``. See :func:`torchtune.models.llama4.llama4_vision_projection_head`
            as an example.
    """

    def __init__(self, clip: nn.Module, projection_head: nn.Module) -> None:
        super().__init__()
        self.clip = clip
        self.projection = projection_head
        register_fusion_module(self.projection)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b x c x w x h]

        Returns:
            Tensor: output tensor of a sequence of embeddings ``[b x s x d]``
                where sequence length (``s``) is ``(num_imgs*num_tiles)+num_embeds``

         Notation used for tensor shapes:
            - b: batch size, equal to flatten(batch x images x tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*clip_embeds_per_tile
            - d: embed dim
        """
        x, _ = self.clip(images[:, None, None])
        x = self.projection(x.squeeze((1, 2)))
        return x
