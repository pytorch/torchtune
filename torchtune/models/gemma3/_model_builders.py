# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torchtune.models.gemma3._component_builders import gemma3, lora_gemma3
from torchtune.modules import TransformerDecoder

from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial

"""
Model builders build specific instantiations using component builders. For example
the ``gemma3_1b`` model builder uses the ``gemma3`` component builder.
"""


def gemma3_1b() -> TransformerDecoder:
    """
    Builder for creating a Gemma3 1B model initialized w/ the default 1b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma3 1B model
    """
    return gemma3(
        vocab_size=262_144,
        num_layers=26,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=1152,
        intermediate_dim=6912,
        # 1B is exception in terms of max_seq_len
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=512,
    )


def lora_gemma3_1b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma3 1B model with LoRA enabled.

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
        TransformerDecoder: Instantiation of Gemma3 1B model with LoRA applied
    """
    return lora_gemma3(
        vocab_size=262_144,
        num_layers=26,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=1152,
        intermediate_dim=6912,
        # 1B is exception in terms of max_seq_len
        max_seq_len=32_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=512,
        # LoRA params
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

qlora_gemma3_1b = partial(lora_gemma3_1b, quantize_base=True)

qlora_gemma3_1b.__doc__ = """
Builder for creating a Gemma3 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma3_1b` for full API arguments.
"""

def gemma3_4b() -> TransformerDecoder:
    """
    Builder for creating a Gemma3 4B model initialized w/ the default 1b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma3 4B model
    """
    return gemma3(
        vocab_size=262_144,
        num_layers=34,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        embed_dim=2560,
        intermediate_dim=10240,
        max_seq_len=128_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=1024,
    )


def lora_gemma3_4b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma3 4B model with LoRA enabled.

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
        TransformerDecoder: Instantiation of Gemma3 4B model with LoRA applied
    """
    return lora_gemma3(
        vocab_size=262_144,
        num_layers=34,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        embed_dim=2560,
        intermediate_dim=10240,
        max_seq_len=128_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        # LoRA params
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        final_capping_value=50.0,
        sliding_window_size=4096,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

qlora_gemma3_4b = partial(lora_gemma3_4b, quantize_base=True)

qlora_gemma3_4b.__doc__ = """
Builder for creating a Gemma3 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma3_4b` for full API arguments.
"""

def gemma3_12b() -> TransformerDecoder:
    """
    Builder for creating a Gemma3 12B model initialized w/ the default 12b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma3 12B model
    """
    return gemma3(
        vocab_size=262_144,
        num_layers=48,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        embed_dim=3840,
        intermediate_dim=15360, # (embed_dim * 8) // 2
        max_seq_len=128_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=1024,
    )


def lora_gemma3_12b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma3 12B model with LoRA enabled.

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
        TransformerDecoder: Instantiation of Gemma3 12B model with LoRA applied
    """
    return lora_gemma3(
        vocab_size=262_144,
        num_layers=48,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        embed_dim=3840,
        intermediate_dim=15360, # (embed_dim * 8) // 2
        max_seq_len=128_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=1024,
        # LoRA params
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

qlora_gemma3_12b = partial(lora_gemma3_1b, quantize_base=True)

qlora_gemma3_12b.__doc__ = """
Builder for creating a Gemma3 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma3_12b` for full API arguments.
"""

def gemma3_27b() -> TransformerDecoder:
    """
    Builder for creating a Gemma3 27B model initialized w/ the default 27b parameter values
    from: https://github.com/google/gemma_pytorch/blob/main/gemma/config.py

    Returns:
        TransformerDecoder: Instantiation of Gemma3 27B model
    """
    return gemma3(
        vocab_size=262_144,
        num_layers=62,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        embed_dim=5376,
        intermediate_dim=86016, # (embed_dim * 8) // 2
        max_seq_len=128_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=1024,
    )


def lora_gemma3_27b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma3 27B model with LoRA enabled.

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
        TransformerDecoder: Instantiation of Gemma3 27B model with LoRA applied
    """
    return lora_gemma3(
        vocab_size=262_144,
        num_layers=62,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        embed_dim=5376,
        intermediate_dim=21504, # (embed_dim * 8) // 2
        max_seq_len=128_000,
        attn_dropout=0.0,
        norm_eps=1e-6,
        sliding_window_size=1024,
        # LoRA params
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

qlora_gemma3_27b = partial(lora_gemma3_1b, quantize_base=True)

qlora_gemma3_27b.__doc__ = """
Builder for creating a Gemma3 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma3_27b` for full API arguments.
"""

