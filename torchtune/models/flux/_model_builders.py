# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torch import nn

from torchtune.models.flux._autoencoder import FluxAutoencoder, FluxDecoder, FluxEncoder
from torchtune.models.flux._flow_model import FluxFlowModel
from torchtune.modules.peft import DoRALinear, LoRALinear


def flux_1_autoencoder(
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
        encoder=encoder,
        decoder=decoder,
    )


def flux_1_dev_flow_model():
    """
    Flow-matching model for FLUX.1-dev

    Returns:
        FluxFlowModel
    """
    return FluxFlowModel(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        use_guidance=True,
    )


def flux_1_schnell_flow_model():
    """
    Flow-matching model for FLUX.1-schnell

    Returns:
        FluxFlowModel
    """
    return FluxFlowModel(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        theta=10_000,
        qkv_bias=True,
        use_guidance=False,
    )


def lora_flux_1_dev_flow_model(
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
):
    """
    Flow-matching model for FLUX.1-dev with linear layers replaced with LoRA modules.

    Args:
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights. Default: False
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
            Default: False

    Returns:
        FluxFlowModel
    """
    model = flux_1_dev_flow_model()
    _replace_linear_with_lora(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
        use_dora=use_dora,
    )
    return model


def lora_flux_1_schnell_flow_model(
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
):
    """
    Flow-matching model for FLUX.1-schnell with linear layers replaced with LoRA modules.

    Args:
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights. Default: False
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
            Default: False

    Returns:
        FluxFlowModel
    """
    model = flux_1_schnell_flow_model()
    _replace_linear_with_lora(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
        use_dora=use_dora,
    )
    return model


def _replace_linear_with_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    quantize_base: bool,
    use_dora: bool,
) -> None:
    lora_cls = DoRALinear if use_dora else LoRALinear
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_child = lora_cls(
                in_dim=child.in_features,
                out_dim=child.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_bias=child.bias is not None,
                quantize_base=quantize_base,
            )
            setattr(module, name, new_child)
        else:
            _replace_linear_with_lora(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                quantize_base=quantize_base,
                use_dora=use_dora,
            )
