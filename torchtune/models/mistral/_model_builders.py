# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torch import nn

from torchtune.models.mistral._component_builders import mistral, lora_mistral

from torchtune.modules import Tokenizer, TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES


"""
Model builders build specific instantiations using component builders. For example
the ``mistral_7b`` model builder uses the ``mistral`` component builder.
"""


def mistral_7b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b parameter values
    from https://mistral.ai/news/announcing-mistral-7b/


    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model
    """
    return mistral(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def mistral_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer


def lora_mistral_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
) -> TransformerDecoder:
    """
    Builder for creating a Llama2 7B model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.llama2_7b`,
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
        max_batch_size (Optional[int]): Maximum batch size to be passed to KVCache.

    Returns:
        TransformerDecoder: Instantiation of Llama2 7B model with LoRA applied
    """
    return lora_mistral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
    )
