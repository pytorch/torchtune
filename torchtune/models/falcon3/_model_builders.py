# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.falcon3._component_builders import lora_falcon3, falcon3
from torchtune.models.falcon3._tokenizer import FALCON3_SPECIAL_TOKENS, Falcon3Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.tokenizers import parse_hf_tokenizer_json

"""
Model builders build specific instantiations using component builders. For example
the falcon3 model builder uses the falcon3 component builder to create the
falcon3 model series.
"""
def falcon3_tokenizer(
    path: str,
    merges_file: str = None,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    **kwargs,
) -> Falcon3Tokenizer:
    """
    Tokenizer for Falcon3.

    Args:
        path (str): path to the vocab.json file.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Falcon3 special tokens.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags. Default is None.

    Returns:
        Falcon3Tokenizer: Instantiation of the Falcon3 tokenizer
    """
    special_tokens = (
        parse_hf_tokenizer_json(special_tokens_path)
        if special_tokens_path is not None
        else FALCON3_SPECIAL_TOKENS
    )
    template = (
        _get_prompt_template(prompt_template) if prompt_template is not None else None
    )
    return Falcon3Tokenizer(
        path=path,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=template,
        **kwargs,
    )


'''
Models
'''
def falcon3_10b() -> TransformerDecoder:
    """
    Builder for creating a Falcon3 model initialized w/ the default 7B parameter values
    from https://huggingface.co/tiiuae/Falcon3-7B

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model
    """
    return falcon3(
        vocab_size=131072,
        num_layers=40,
        num_heads=12,
        num_kv_heads=4,
        embed_dim=3072,
        intermediate_dim=23040,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
    )


def falcon3_7b() -> TransformerDecoder:
    """
    Builder for creating a Falcon3 model initialized w/ the default 7B parameter values
    from https://huggingface.co/tiiuae/Falcon3-7B

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model
    """
    return falcon3(
        vocab_size=131072,
        num_layers=28,
        num_heads=12,
        num_kv_heads=4,
        embed_dim=3072,
        intermediate_dim=23040,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
    )



def falcon3_3b() -> TransformerDecoder:
    """
    Builder for creating a Falcon3 model initialized w/ the default 7B parameter values
    from https://huggingface.co/tiiuae/Falcon3-7B

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model
    """
    return falcon3(
        vocab_size=131072,
        num_layers=22,
        num_heads=12,
        num_kv_heads=4,
        embed_dim=3072,
        intermediate_dim=9216,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
    )


def falcon3_1b() -> TransformerDecoder:
    """
    Builder for creating a Falcon3 model initialized w/ the default 7B parameter values
    from https://huggingface.co/tiiuae/Falcon3-7B

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model
    """
    return falcon3(
        vocab_size=131072,
        num_layers=18,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=8192,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
    )
'''
LoRA 
'''

def lora_falcon3_10b(
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
    Builder for creating a Falcon3 7B model with LoRA enabled.

    The Falcon3 defaults are the same as in :func:`~torchtune.models.falcon3.falcon3_7b`,
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

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model with LoRA applied
    """
    return lora_falcon3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=131072,
        num_layers=40,
        num_heads=12,
        num_kv_heads=4,
        embed_dim=3072,
        intermediate_dim=23040,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_falcon3_7b(
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
    Builder for creating a Falcon3 7B model with LoRA enabled.

    The Falcon3 defaults are the same as in :func:`~torchtune.models.falcon3.falcon3_7b`,
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

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model with LoRA applied
    """
    return lora_falcon3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=131072,
        num_layers=28,
        num_heads=12,
        num_kv_heads=4,
        embed_dim=3072,
        intermediate_dim=23040,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1_000_042,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_falcon3_3b(
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
    Builder for creating a Falcon3 7B model with LoRA enabled.

    The Falcon3 defaults are the same as in :func:`~torchtune.models.falcon3.falcon3_7b`,
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

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model with LoRA applied
    """
    return lora_falcon3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=131072,
        num_layers=22,
        num_heads=12,
        num_kv_heads=4,
        embed_dim=3072,
        intermediate_dim=9216,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_falcon3_1b(
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
    Builder for creating a Falcon3 7B model with LoRA enabled.

    The Falcon3 defaults are the same as in :func:`~torchtune.models.falcon3.falcon3_7b`,
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

    Returns:
        TransformerDecoder: Instantiation of Falcon3 7B model with LoRA applied
    """
    return lora_falcon3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=131072,
        num_layers=18,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=8192,
        max_seq_len=4096,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1_000_042,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
