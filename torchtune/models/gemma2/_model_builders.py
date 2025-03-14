# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import List

from torchtune.models.gemma2._component_builders import gemma2, lora_gemma2
from torchtune.modules import TransformerDecoder

from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using component builders. For example
the ``gemma_2b`` model builder uses the ``gemma2`` component builder.
"""


def gemma2_2b() -> TransformerDecoder:
    """
    Builder for creating a Gemma2 2B model initialized w/ the default 2b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma2 2B model
    """
    return gemma2(
        vocab_size=256_000,
        num_layers=26,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        embed_dim=2304,
        intermediate_dim=9216,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        hidden_capping_value=30.0,
        final_capping_value=50.0,
        sliding_window_size=4096,
    )


def lora_gemma2_2b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma2 2B model with LoRA enabled.

    The Gemma defaults are the same as in :func:`~torchtune.models.gemma.gemma_2b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Gemma2 2B model with LoRA applied
    """
    return lora_gemma2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
        num_layers=26,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        embed_dim=2304,
        intermediate_dim=9216,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        hidden_capping_value=30.0,
        final_capping_value=50.0,
        sliding_window_size=4096,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_gemma2_2b = partial(lora_gemma2_2b, quantize_base=True)

qlora_gemma2_2b.__doc__ = """
Builder for creating a Gemma2 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma2_2b` for full API arguments.
"""


def gemma2_9b() -> TransformerDecoder:
    """
    Builder for creating a Gemma2 9B model initialized w/ the default 9b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma 9B model
    """
    return gemma2(
        vocab_size=256_000,
        num_layers=42,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        embed_dim=3584,
        intermediate_dim=14336,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        hidden_capping_value=30.0,
        final_capping_value=50.0,
        sliding_window_size=4096,
    )


def lora_gemma2_9b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma 9B model with LoRA enabled.

    The Gemma defaults are the same as in :func:`~torchtune.models.gemma.gemma_7b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Gemma2 9B model with LoRA applied
    """
    return lora_gemma2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
        num_layers=42,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        embed_dim=3584,
        intermediate_dim=14336,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        hidden_capping_value=30.0,
        final_capping_value=50.0,
        sliding_window_size=4096,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_gemma2_9b = partial(lora_gemma2_9b, quantize_base=True)

qlora_gemma2_9b.__doc__ = """
Builder for creating a Gemma model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma2_9b` for full API arguments.
"""


def gemma2_27b() -> TransformerDecoder:
    """
    Builder for creating a Gemma2 27B model initialized w/ the default 27b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma2 27B model
    """
    return gemma2(
        vocab_size=256_000,
        num_layers=46,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        embed_dim=4608,
        intermediate_dim=36864,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        hidden_capping_value=30.0,
        final_capping_value=50.0,
        sliding_window_size=4096,
        query_pre_attn_scalar=144,
    )


def lora_gemma2_27b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma2 27B model with LoRA enabled.

    The Gemma defaults are the same as in :func:`~torchtune.models.gemma.gemma_7b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Gemma2 27B model with LoRA applied
    """
    return lora_gemma2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
        num_layers=46,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        embed_dim=4608,
        intermediate_dim=36864,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        hidden_capping_value=30.0,
        final_capping_value=50.0,
        sliding_window_size=4096,
        query_pre_attn_scalar=144,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_gemma2_27b = partial(lora_gemma2_27b, quantize_base=True)

qlora_gemma2_27b.__doc__ = """
Builder for creating a Gemma model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma2_27b` for full API arguments.
"""
