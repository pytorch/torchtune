# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import List, Optional

from torchtune.data._prompt_templates import (_get_prompt_template,
                                              _TemplateType)
from torchtune.models.llama3._component_builders import llama3, lora_llama3
from torchtune.models.sarvam1._tokenizer import Sarvam1Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using Llama 3 component builders.
"""


def sarvam1_tokenizer(
    path: str,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[
        _TemplateType
    ] = "torchtune.models.sarvam1.Sarvam1ChatTemplate",
) -> Sarvam1Tokenizer:
    """
    Tokenizer for Sarvam1.

    Args:
        path (str): path to the tokenizer
        max_seq_len (Optional[int]): max sequence length to truncate tokens to.
        prompt_template (Optional[str]): optional specified prompt template.
            If given, assumed to be a huggingface prompt template name.

    Returns:
        Sarvam1Tokenizer: Instantiation of the Llama2 tokenizer
    """
    return Sarvam1Tokenizer(
        path=path,
        max_seq_len=max_seq_len,
        prompt_template=(
            _get_prompt_template(prompt_template)
            if prompt_template is not None
            else None
        ),
    )


def sarvam1() -> TransformerDecoder:
    """
    Builder for creating a Sarvam1 model initialized w/ the default parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3 8B model
    """
    return llama3(
        vocab_size=68096,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=8192,
        intermediate_dim=11008,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=10000,
    )


def lora_sarvam1(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Sarvam1 model with LoRA enabled.

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
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).

    Returns:
        TransformerDecoder: Instantiation of Llama3 8B model with LoRA applied
    """
    return lora_llama3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=68096,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=8192,
        intermediate_dim=11008,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=10000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
        use_dora=use_dora,
    )
