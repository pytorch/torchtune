# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from functools import partial

from torchtune.models.llama3_1._component_builders import llama3_1, lora_llama3_1

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using component builders. For example
the llama3_1_8b model builder uses the llama3 component builder to create the
Llama3.1 8B model.
"""


def llama3_1_8b() -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 model initialized w/ the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 8B model
    """
    return llama3_1(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131072,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )


def llama3_1_70b() -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 model initialized w/ the default 70B parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 70B model
    """
    return llama3_1(
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        max_seq_len=131072,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )


def llama3_1_405b() -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 model initialized w/ the default 405B parameter values.
    
    Returns:
        TransformerDecoder: Instantiation of Llama3.1 405B model
    """
    return llama3_1(
        vocab_size=128_256,
        num_layers=126,
        num_heads=128,
        num_kv_heads=8,
        embed_dim=16384,
        max_seq_len=8192,
        intermediate_dim=53248,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
    )


def lora_llama3_1_8b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 8B model with LoRA enabled.

    The Llama3.1 defaults are the same as in :func:`~torchtune.models.llama3_1.llama3_1_8b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 8B model with LoRA applied
    """
    return lora_llama3_1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131072,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_llama3_1_70b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 70B model with LoRA enabled.

    The Llama3.1 defaults are the same as in :func:`~torchtune.models.llama3_1.llama3_1_70b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 70B model with LoRA applied
    """
    return lora_llama3_1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        max_seq_len=131072,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_llama3_1_405b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.1 405B model with LoRA enabled.

    The Llama3.1 defaults are the same as in :func:`~torchtune.models.llama3.llama3_8b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3.1 8B model with LoRA applied
    """
    return lora_llama3_1(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=126,
        num_heads=128,
        num_kv_heads=8,
        embed_dim=16384,
        max_seq_len=8192,
        intermediate_dim=53248,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
    )


qlora_llama3_1_8b = partial(lora_llama3_1_8b, quantize_base=True)

qlora_llama3_1_8b.__doc__ = """
Builder for creating a Llama3.1 8B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_1_8b` for full API arguments.
"""

qlora_llama3_1_70b = partial(lora_llama3_1_70b, quantize_base=True)

qlora_llama3_1_70b.__doc__ = """
Builder for creating a Llama3.1 70B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_1_70b` for full API arguments.
"""

qlora_llama3_1_405b = partial(lora_llama3_1_405b, quantize_base=True)

qlora_llama3_1_405b.__doc__ = """
Builder for creating a Llama3.1 405B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_1_405b` for full API arguments.
"""
