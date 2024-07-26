# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from functools import partial

from torchtune.models.llama2._component_builders import llama2, lora_llama2

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES


def code_llama2_7b() -> TransformerDecoder:
    """
    Builder for creating a Code-Llama2 model initialized w/ the default 7B parameter values
    from https://arxiv.org/pdf/2308.12950.pdf

    Returns:
        TransformerDecoder: Instantiation of Code-Llama2 7B model
    """
    return llama2(
        vocab_size=32_016,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_code_llama2_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Code-Llama2 7B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.code_llama2_7b`,
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
        lora_dropout (float): dropout probability for LoRA linear layers. Default: 0.05
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Code-Llama2 7B model with LoRA applied
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_016,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
    )


qlora_code_llama2_7b = partial(lora_code_llama2_7b, quantize_base=True)

qlora_code_llama2_7b.__doc__ = """
Builder for creating a Code-Llama2 7B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_code_llama2_7b` for full API arguments.
"""


def code_llama2_13b() -> TransformerDecoder:
    """
    Builder for creating a Code-Llama2 model initialized w/ the default 13B parameter values
    from https://arxiv.org/pdf/2308.12950.pdf

    Returns:
        TransformerDecoder: Instantiation of Code-Llama2 13B model
    """
    return llama2(
        vocab_size=32_016,
        num_layers=40,
        num_heads=40,
        num_kv_heads=40,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_code_llama2_13b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Code-Llama2 13B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.code_llama2_13b`,
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
        lora_dropout (float): dropout probability for LoRA linear layers. Default: 0.05
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Code-Llama2 13B model with LoRA applied
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_016,
        num_layers=40,
        num_heads=40,
        num_kv_heads=40,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
    )


qlora_code_llama2_13b = partial(lora_code_llama2_13b, quantize_base=True)

qlora_code_llama2_13b.__doc__ = """
Builder for creating a Code-Llama2 13B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_code_llama2_13b` for full API arguments.
"""


def code_llama2_70b() -> TransformerDecoder:
    """
    Builder for creating a Code-Llama2 model initialized w/ the default 70B parameter values
    from https://arxiv.org/pdf/2308.12950.pdf

    Returns:
        TransformerDecoder: Instantiation of Code-Llama2 70B model
    """
    return llama2(
        vocab_size=32_016,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=28672,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_code_llama2_70b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.05,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Code-Llama2 70B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.code_llama2_70b`,
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
        lora_dropout (float): dropout probability for LoRA linear layers. Default: 0.05
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Code-Llama2 70B model with LoRA applied
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_016,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=28672,
        max_seq_len=16384,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
    )


qlora_code_llama2_70b = partial(lora_code_llama2_70b, quantize_base=True)

qlora_code_llama2_70b.__doc__ = """
Builder for creating a Code-Llama2 70B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_code_llama2_70b` for full API arguments.
"""
