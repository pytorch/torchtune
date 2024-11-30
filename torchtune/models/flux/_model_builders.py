# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.models.flux._autoencoder import FluxAutoencoder


def flux_autoencoder() -> FluxAutoencoder:
    """
    The image autoencoder for all current Flux diffusion models:
    - FLUX.1-dev
    - FLUX.1-schnell
    - FLUX.1-Canny-dev
    - FLUX.1-Depth-dev
    - FLUX.1-Fill-dev

    Returns:
        FluxAutoencoder
    """
    # ch = number of channels (size of the channel dimension)
    return FluxAutoencoder(
        resolution=256,
        ch_in=3,
        ch_out=3,
        ch_base=128,
        ch_mults=[1, 2, 4, 4],
        ch_z=16,
        n_layers_per_resample_block=2,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
