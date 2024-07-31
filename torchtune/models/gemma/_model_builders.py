# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

from torchtune.models.gemma._component_builders import gemma, lora_gemma
from torchtune.models.gemma.transformer import GemmaTransformerDecoder

from torchtune.models.gemma._tokenizer import GemmaTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES

from functools import partial

"""
Model builders build specific instantiations using component builders. For example
the ``gemma_2b`` model builder uses the ``gemma`` component builder.
"""


def gemma_2b() -> GemmaTransformerDecoder:
    """
    Builder for creating a Gemma 2B model initialized w/ the default 2b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model
    """
    return gemma(
        vocab_size=256_000,
        num_layers=18,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=2048,
        intermediate_dim=16384,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
    )


def gemma_tokenizer(path: str) -> GemmaTokenizer:
    """
    Tokenizer for Gemma.

    Args:
        path (str): path to the tokenizer

    Returns:
        GemmaTokenizer: Instantiation of the Gemma tokenizer
    """
    return GemmaTokenizer(path)


def lora_gemma_2b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> GemmaTransformerDecoder:
    """
    Builder for creating a Gemma 2B model with LoRA enabled.

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
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 2B model with LoRA applied
    """
    return lora_gemma(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
        num_layers=18,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        embed_dim=2048,
        intermediate_dim=16384,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )

qlora_gemma_2b = partial(lora_gemma_2b, quantize_base=True)

qlora_gemma_2b.__doc__ = """
Builder for creating a Gemma model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma_2b` for full API arguments.
"""



def gemma_7b() -> GemmaTransformerDecoder:
    """
    Builder for creating a Gemma 7B model initialized w/ the default 7b parameter values
    from: https://blog.google/technology/developers/gemma-open-models/

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 7B model
    """
    return gemma(
        vocab_size=256_000,
        num_layers=28,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        embed_dim=3072,
        intermediate_dim=24576,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
    )
    
    
def lora_gemma_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    quantize_base: bool = False,
) -> GemmaTransformerDecoder:
    """
    Builder for creating a Gemma 7B model with LoRA enabled.

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
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        GemmaTransformerDecoder: Instantiation of Gemma 7B model with LoRA applied
    """
    return lora_gemma(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        vocab_size=256_000,
        num_layers=28,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        embed_dim=3072,
        intermediate_dim=24576,
        max_seq_len=8192,
        attn_dropout=0.0,
        norm_eps=1e-6,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        quantize_base=quantize_base,
    )

qlora_gemma_7b = partial(lora_gemma_7b, quantize_base=True)

qlora_gemma_7b.__doc__ = """
Builder for creating a Gemma model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_gemma_7b` for full API arguments.
"""
