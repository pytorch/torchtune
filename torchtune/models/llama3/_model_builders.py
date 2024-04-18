# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torch import nn

from torchtune.models.llama3._component_builders import llama3, lora_llama3
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import TikTokenTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES


"""
Model builders build specific instantiations using component builders. For example
the llama3_8b model builder uses the llama3 component builder to create the
Llama3 8B model.
"""


def llama3_8b() -> TransformerDecoder:
    """
    Builder for creating a Llama3 model initialized w/ the default 8b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3 8B model
    """
    return llama3(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=4096,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )


def llama3_tokenizer(path: str) -> TikTokenTokenizer:
    tiktoken = TikTokenTokenizer(path)
    tiktoken.pad_id = 0
    return tiktoken


def lora_llama3_8b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama3 8B model with LoRA enabled.

    The Llama3 defaults are the same as in :func:`~torchtune.models.llama3.llama3_8b`,
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
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3 8B model with LoRA applied
    """
    return lora_llama3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=4096,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )

qlora_llama3_8b = partial(lora_llama3_8b, quantize_base=True)

qlora_llama3_8b.__doc__ = """
Builder for creating a Llama3 model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_8b` for full API arguments.
"""
