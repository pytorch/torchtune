# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.models.flux._autoencoder import FluxAutoencoder, FluxDecoder, FluxEncoder


def flux_1_autoencoder(
    resolution: int = 256,
    ch_in: int = 3,
    ch_out: int = 3,
    ch_base: int = 128,
    ch_mults: List[int] = [1, 2, 4, 4],
    ch_z: int = 16,
    n_layers_per_resample_block: int = 2,
    scale_factor: float = 0.3611,
    shift_factor: float = 0.1159,
) -> FluxAutoencoder:
    """
    The image autoencoder for all current Flux diffusion models:
    - FLUX.1-dev
    - FLUX.1-schnell
    - FLUX.1-Canny-dev
    - FLUX.1-Depth-dev
    - FLUX.1-Fill-dev

    ch = number of channels (size of the channel dimension)

    Args:
        resolution (int): The height/width of the square input image.
        ch_in (int): The number of channels of the input image.
        ch_out (int): The number of channels of the output image.
        ch_base (int): The base number of channels.
            This gets multiplied by `ch_mult` values to get the number of inner channels during downsampling/upsampling.
        ch_mults (List[int]): The channel multiple per downsample/upsample block.
            This gets multiplied by `ch_base` to get the number of inner channels during downsampling/upsampling.
        ch_z (int): The number of latent channels (dimension of the latent vector `z`).
        n_layers_per_resample_block (int): Number of resnet layers per downsample/upsample block.
        scale_factor (float): Constant for scaling `z`.
        shift_factor (float): Constant for shifting `z`.

    Returns:
        FluxAutoencoder
    """
    channels = [ch_base * mult for mult in ch_mults]

    encoder = FluxEncoder(
        ch_in=ch_in,
        ch_z=ch_z,
        channels=channels,
        n_layers_per_down_block=n_layers_per_resample_block,
        scale_factor=scale_factor,
        shift_factor=shift_factor,
    )

    decoder = FluxDecoder(
        ch_out=ch_out,
        ch_z=ch_z,
        channels=list(reversed(channels)),
        # decoder gets one more layer per up block than the encoder's down blocks
        n_layers_per_up_block=n_layers_per_resample_block + 1,
        scale_factor=scale_factor,
        shift_factor=shift_factor,
    )

    return FluxAutoencoder(
        img_shape=(ch_in, resolution, resolution),
        encoder=encoder,
        decoder=decoder,
    )
