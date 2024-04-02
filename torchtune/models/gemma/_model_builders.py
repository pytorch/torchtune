# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torch import nn

from torchtune.models.gemma._component_builders import gemma, lora_gemma

from torchtune.modules import Tokenizer, TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using component builders. For example
the ``gemma_2b`` model builder uses the ``gemma`` component builder.
"""


def gemma_2b() -> TransformerDecoder:
    """
    Builder for creating a Gemma 2B model initialized w/ the default 2b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        TransformerDecoder: Instantiation of Gemma 2B model
    """
    return gemma(
        vocab_size=256_000,
        num_layers=18,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=2048,
        intermediate_dim=16384,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
    )


def gemma_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    tokenizer.pad_id = 0
    return tokenizer


def lora_gemma_2b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
) -> TransformerDecoder:
    """
    Builder for creating a Gemma 2B model with LoRA enabled.

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

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model with LoRA applied
    """
    model = lora_gemma(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        # model parameters
        vocab_size=256_000,
        num_layers=18,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=2048,
        intermediate_dim=16384,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        # lora parameters
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
    )
    return model
