# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torchtune.models.llama3._component_builders import llama3, lora_llama3

from torchtune.modules import TransformerDecoder
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.tokenizers import parse_hf_tokenizer_json
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._prompt_templates import _get_prompt_template


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
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )


def llama3_70b() -> TransformerDecoder:
    """
    Builder for creating a Llama3 model initialized w/ the default 70B parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3 70 model
    """
    return llama3(
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        max_seq_len=8192,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
    )

 
def llama3_tokenizer(path: str, special_tokens_path: Optional[str] = None, max_seq_len: Optional[int] = None, prompt_template: Optional[_TemplateType] = None) -> Llama3Tokenizer:
    """
    Tokenizer for Llama3.

    Args:
        path (str): path to the tokenizer
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file 
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
    
    Returns:
        Llama3Tokenizer: Instantiation of the Llama3 tokenizer
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return Llama3Tokenizer(path=path, special_tokens=special_tokens, max_seq_len=max_seq_len, prompt_template=template)


def lora_llama3_8b(
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
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
        use_dora=use_dora,
    )


def lora_llama3_70b(
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
    Builder for creating a Llama3 70B model with LoRA enabled.

    The Llama3 defaults are the same as in :func:`~torchtune.models.llama3.llama3_70b`,
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
        TransformerDecoder: Instantiation of Llama3 70B model with LoRA applied
    """
    return lora_llama3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        max_seq_len=8192,
        intermediate_dim=28672,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
        use_dora=use_dora,
    )


qlora_llama3_8b = partial(lora_llama3_8b, quantize_base=True)

qlora_llama3_8b.__doc__ = """
Builder for creating a Llama3 8B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_8b` for full API arguments.
"""

qlora_llama3_70b = partial(lora_llama3_70b, quantize_base=True)

qlora_llama3_70b.__doc__ = """
Builder for creating a Llama3 70B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_70b` for full API arguments.
"""
