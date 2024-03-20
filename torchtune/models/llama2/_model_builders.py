# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torch import nn

from torchtune.models.llama2._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama2._component_builders import llama2, lora_llama2

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    KVCache,
    RMSNorm,
    RotaryPositionalEmbeddings,
    Tokenizer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torchtune.modules.peft import LoRALinear, LORA_ATTN_MODULES


"""
Model builders build specific instantiations using component builders. For example
the llama2_7b model builder uses the llama2 component builder to create the
llama2 7B model.
"""


def llama2_7b(max_batch_size: Optional[int] = None) -> TransformerDecoder:
    """Builder for creating a Llama2 model initialized w/ the default 7b parameter values.
    From https://arxiv.org/abs/2307.09288, these default values are:
    - vocab_size: 32,000
    - embed_dim: 4,096
    - num_layers: 32
    - num_heads: 32
    - num_kv_heads: 32
    - max_seq_len: 4,096
    - norm_eps: 1e-5

    Args:
        max_batch_size (Optional[int]): Maximum batch size to be passed to KVCache.

    Returns:
        A ``TransformerDecoder`` instance of the Llama2 model.
    """
    return llama2(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        max_batch_size=max_batch_size,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def llama2_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer


def lora_llama2_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    max_batch_size: Optional[int] = None,
) -> TransformerDecoder:
    """Builder for creating a Llama2 model with LoRA enabled.

    The Llama2 defaults are the same as in :func:`~torchtune.models.llama2.llama2_7b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        max_batch_size=max_batch_size,
        attn_dropout=0.0,
        norm_eps=1e-5,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
    )
