# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.models.mistral._component_builders import (
    mistral,
    lora_mistral,
    mistral_classifier,
    lora_mistral_classifier,
)

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial


def mistral_24b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 24B model initialized w/ the default 24b parameter values
    from https://mistral.ai/en/news/mistral-small-3
    """
    return mistral(
        vocab_size=131_072,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=32768,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000_000,
    )


def lora_mistral_24b(
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
    Builder for creating a Mistral 24B model with LoRA enabled.

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
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral 24B model with LoRA applied
    """
    return lora_mistral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=131_072,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=32768,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_mistral_24b = partial(lora_mistral_24b, quantize_base=True)

qlora_mistral_24b.__doc__ = """
Builder for creating a Mistral 24B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_mistral_24b` for full API arguments.
"""


def mistral_24b_reward() -> TransformerDecoder:
    """
    Builder for creating a Mistral 24B model initialized w/ the default 24b parameter values
    from https://mistral.ai/en/news/mistral-small-3
    """
    return mistral_classifier(
        num_classes=1,
        vocab_size=131_072,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=32768,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000_000,
    )


def lora_mistral_24b_reward(
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
    Builder for creating a Mistral 24B reward model with LoRA enabled.
    """
    return lora_mistral_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=131_072,
        num_layers=40,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=32768,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=100_000_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_mistral_24b_reward = partial(lora_mistral_24b_reward, quantize_base=True)

qlora_mistral_24b_reward.__doc__ = """
Builder for creating a Mistral 24B reward model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_mistral_24b_reward` for full API arguments.
"""
