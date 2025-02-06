# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
from contextlib import nullcontext
from typing import Callable, ContextManager, Optional

import torch
from torch import Tensor

from torchtune.models.flux._autoencoder import FluxDecoder
from torchtune.models.flux._flow_model import FluxFlowModel

PATCH_HEIGHT, PATCH_WIDTH = 2, 2
POSITION_DIM = 3
LATENT_CHANNELS = 16
IMG_LATENT_SIZE_RATIO = 8


def predict_noise(
    model: FluxFlowModel,
    latents: Tensor,
    clip_encodings: Tensor,
    t5_encodings: Tensor,
    timesteps: Tensor,
    guidance: Optional[Tensor] = None,
    model_ctx: ContextManager = nullcontext(),
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
        model_ctx (ContextManager): Optional context to wrap the model call (e.g. for activation offloading)
            Default: nullcontext

    Returns:
        Tensor: The noise prediction.
            Shape: [bsz, 16, latent height, latent width]
    """
    bsz, _, latent_height, latent_width = latents.shape

    with torch.no_grad():
        # Create positional encodings
        latent_pos_enc = create_position_encoding_for_latents(
            bsz, latent_height, latent_width
        )
        text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

        # Convert latent into a sequence of patches
        latents = pack_latents(latents)

    # Predict noise
    with model_ctx:
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


def get_t5_max_seq_len(flux_model_name: str) -> int:
    """
    Get the maximum sequence length of the T5 tokenizer for the given Flux model.

    Args:
        flux_model_name (str): "FLUX.1-dev" or "FLUX.1-schnell"

    Returns:
        int: The T5 max seq len.

    Raises:
        ValueError: if flux model name is invalid
    """
    if flux_model_name == "FLUX.1-dev":
        return 512
    if flux_model_name == "FLUX.1-schnell":
        return 256
    raise ValueError(f"Unknown Flux model: {flux_model_name}")


def create_noise(
    bsz: int,
    img_height: int,
    img_width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
):
    return torch.randn(
        bsz,
        LATENT_CHANNELS,
        img_height // IMG_LATENT_SIZE_RATIO,
        img_width // IMG_LATENT_SIZE_RATIO,
        device=device,
        dtype=dtype,
        generator=(
            None if seed is None else torch.Generator(device=device).manual_seed(seed)
        ),
    )


def generate_images(
    model: FluxFlowModel,
    decoder: FluxDecoder,
    clip_encodings: Tensor,
    t5_encodings: Tensor,
    guidance: Tensor,
    img_height: int,
    img_width: int,
    seed: int,
    denoising_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    # create initial latents
    bsz = clip_encodings.shape[0]
    latents = create_noise(bsz, img_height, img_width, device, dtype, seed)
    _, latent_channels, latent_height, latent_width = latents.shape

    # create denoising schedule
    timesteps = _get_timesteps(denoising_steps, latent_channels, shift=True)

    # create positional encodings
    latent_pos_enc = create_position_encoding_for_latents(
        bsz, latent_height, latent_width
    ).to(latents)
    text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM).to(latents)

    # convert img-like latents into sequences of patches
    latents = pack_latents(latents)

    # iteratively denoise latents
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((bsz,), t_curr, dtype=dtype, device=device)
        pred = model(
            img=latents,
            img_ids=latent_pos_enc,
            txt=t5_encodings,
            txt_ids=text_pos_enc,
            y=clip_encodings,
            timesteps=t_vec,
            guidance=guidance,
        )
        latents = latents + (t_prev - t_curr) * pred

    # convert sequences of patches into img-like latents
    latents = unpack_latents(latents, latent_height, latent_width)

    # decode latents into images
    imgs = decoder(latents)
    return imgs


def _get_timesteps(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = _get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = _time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def _get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
