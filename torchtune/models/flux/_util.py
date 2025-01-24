# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import Tensor

from torchtune.models.flux._flow_model import FluxFlowModel

PATCH_HEIGHT, PATCH_WIDTH = 2, 2
POSITION_DIM = 3


def predict_noise(
    model: FluxFlowModel,
    latents: Tensor,
    clip_encodings: Tensor,
    t5_encodings: Tensor,
    timesteps: Tensor,
    guidance: Optional[Tensor] = None,
) -> Tensor:
    """
    Use Flux's flow-matching model to predict the noise in image latents.

    Args:
        model (FluxFlowModel): The Flux flow model.
        latents (Tensor): Image encodings from the Flux autoencoder.
            Shape: [bsz, 16, latent height, latent width]
        clip_encodings (Tensor): CLIP text encodings.
            Shape: [bsz, 768]
        t5_encodings (Tensor): T5 text encodings.
            Shape: [bsz, sequence length, 256 or 512]
        timesteps (Tensor): The amount of noise (0 to 1).
            Shape: [bsz]
        guidance (Optional[Tensor]): The guidance value (1.5 to 4) if guidance-enabled model.
            Shape: [bsz]
            Default: None

    Returns:
        Tensor: The noise prediction.
            Shape: [bsz, 16, latent height, latent width]
    """
    bsz, _, latent_height, latent_width = latents.shape

    # Create positional encodings
    latent_pos_enc = create_position_encoding_for_latents(
        bsz, latent_height, latent_width
    )
    text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

    # Convert latent into a sequence of patches
    latents = pack_latents(latents)

    # Predict noise
    latent_noise_pred = model(
        img=latents,
        img_ids=latent_pos_enc.to(latents),
        txt=t5_encodings.to(latents),
        txt_ids=text_pos_enc.to(latents),
        y=clip_encodings.to(latents),
        timesteps=timesteps.to(latents),
        guidance=guidance.to(latents) if guidance is not None else None,
    )

    # Convert sequence of patches to latent shape
    latent_noise_pred = unpack_latents(latent_noise_pred, latent_height, latent_width)

    return latent_noise_pred


def create_position_encoding_for_latents(
    bsz: int, latent_height: int, latent_width: int
) -> Tensor:
    """
    Create the packed latents' position encodings for the Flux flow model.

    Args:
        bsz (int): The batch size.
        latent_height (int): The height of the latent.
        latent_width (int): The width of the latent.

    Returns:
        Tensor: The position encodings.
            Shape: [bsz, (latent_height // PATCH_HEIGHT) * (latent_width // PATCH_WIDTH), POSITION_DIM)
    """
    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH

    position_encoding = torch.zeros(height, width, POSITION_DIM)

    row_indices = torch.arange(height)
    position_encoding[:, :, 1] = row_indices.unsqueeze(1)

    col_indices = torch.arange(width)
    position_encoding[:, :, 2] = col_indices.unsqueeze(0)

    # Flatten and repeat for the full batch
    # [height, width, 3] -> [bsz, height * width, 3]
    position_encoding = position_encoding.view(1, height * width, POSITION_DIM)
    position_encoding = position_encoding.repeat(bsz, 1, 1)

    return position_encoding


def pack_latents(x: Tensor) -> Tensor:
    """
    Rearrange latents from an image-like format into a sequence of patches.

    Equivalent to `einops.rearrange("b c (h ph) (w pw) -> b (h w) (c ph pw)")`.

    Args:
        x (Tensor): The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]

    Returns:
        Tensor: The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
    """
    b, c, latent_height, latent_width = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH

    # [b, c, h*ph, w*ph] -> [b, c, h, w, ph, pw]
    x = x.unfold(2, PATCH_HEIGHT, PATCH_HEIGHT).unfold(3, PATCH_WIDTH, PATCH_WIDTH)

    # [b, c, h, w, ph, PW] -> [b, h, w, c, ph, PW]
    x = x.permute(0, 2, 3, 1, 4, 5)

    # [b, h, w, c, ph, PW] -> [b, h*w, c*ph*PW]
    return x.reshape(b, h * w, c * PATCH_HEIGHT * PATCH_WIDTH)


def unpack_latents(x: Tensor, latent_height: int, latent_width: int) -> Tensor:
    """
    Rearrange latents from a sequence of patches into an image-like format.

    Equivalent to `einops.rearrange("b (h w) (c ph pw) -> b c (h ph) (w pw)")`.

    Args:
        x (Tensor): The packed latents.
            Shape: (bsz, (latent_height // ph) * (latent_width // pw), ch * ph * pw)
        latent_height (int): The height of the unpacked latents.
        latent_width (int): The width of the unpacked latents.

    Returns:
        Tensor: The unpacked latents.
            Shape: [bsz, ch, latent height, latent width]
    """
    b, _, c_ph_pw = x.shape
    h = latent_height // PATCH_HEIGHT
    w = latent_width // PATCH_WIDTH
    c = c_ph_pw // (PATCH_HEIGHT * PATCH_WIDTH)

    # [b, h*w, c*ph*pw] -> [b, h, w, c, ph, pw]
    x = x.reshape(b, h, w, c, PATCH_HEIGHT, PATCH_WIDTH)

    # [b, h, w, c, ph, pw] -> [b, c, h, ph, w, pw]
    x = x.permute(0, 3, 1, 4, 2, 5)

    # [b, c, h, ph, w, pw] -> [b, c, h*ph, w*pw]
    return x.reshape(b, c, h * PATCH_HEIGHT, w * PATCH_WIDTH)
