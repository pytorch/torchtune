# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.models.qwen2._component_builders import qwen2, lora_qwen2
from torchtune.models.qwen2._tokenizer import Qwen2Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.tokenizers import parse_hf_tokenizer_json
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._prompt_templates import _get_prompt_template

"""
Model builders build specific instantiations using component builders. For example
the qwen2_7b model builder uses the qwen2 component builder to create the
qwen2 7B model.
"""


def qwen2_7b() -> TransformerDecoder:
    """
    Builder for creating a Qwen2 model initialized w/ the default 7B parameter values
    from https://huggingface.co/Qwen/Qwen2-7B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2 7B model
    """
    return qwen2(
        vocab_size=152064,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        embed_dim=3584,
        intermediate_dim=18944,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1000000.0,
    )


def qwen2_0_5b() -> TransformerDecoder:
    """
    Builder for creating a Qwen2 model initialized w/ the default 0.5B parameter values
    from https://huggingface.co/Qwen/Qwen2-0.5B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2 0.5B model

    Note:
        Qwen2 0.5B and Qwen2 1.5B model builders will enable `tie_word_embeddings` by default
        and returns an instance of `TransformerDecoder`.
    """
    return qwen2(
        vocab_size=151936,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        embed_dim=896,
        intermediate_dim=4864,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_1_5b() -> TransformerDecoder:
    """
    Builder for creating a Qwen2 model initialized w/ the default 1.5B parameter values
    from https://huggingface.co/Qwen/Qwen2-1.5B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2 1.5B model

    Note:
        Qwen2 0.5B and Qwen2 1.5B model builders will enable `tie_word_embeddings` by default
        and returns an instance of `TransformerDecoder`.
    """
    return qwen2(
        vocab_size=151936,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        embed_dim=1536,
        intermediate_dim=8960,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-06,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_tokenizer(
    path: str,
    merges_file: str = None,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = "torchtune.data.ChatMLTemplate",
    **kwargs,
) -> Qwen2Tokenizer:
    """
    Tokenizer for Qwen2.

    Args:
        path (str): path to the vocab.json file.
        merges_file (str): path to the merges.txt file.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen2 special tokens.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags. Default is :class:`~torchtune.models.llama2.Llama2ChatTemplate`.

    Returns:
        Qwen2Tokenizer: Instantiation of the Qwen2 tokenizer
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    template = _get_prompt_template(prompt_template) if prompt_template is not None else None
    return Qwen2Tokenizer(path=path, merges_file=merges_file, special_tokens=special_tokens, max_seq_len=max_seq_len, prompt_template=template, **kwargs)


def lora_qwen2_7b(
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
    Builder for creating a Qwen2 7B model with LoRA enabled.

    The Qwen2 defaults are the same as in :func:`~torchtune.models.qwen2.qwen2_7b`,
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
        TransformerDecoder: Instantiation of Qwen2 7B model with LoRA applied
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        embed_dim=3584,
        intermediate_dim=18944,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen2_0_5b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Qwen2 0.5B model with LoRA enabled.

    The Qwen2 defaults are the same as in :func:`~torchtune.models.qwen2.qwen2_0_5b`,
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
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Qwen2 0.5B model with LoRA applied

    Note:
        Qwen2 0.5B and Qwen2 1.5B model builders will enable `tie_word_embeddings` by default
        and returns an instance of `TransformerDecoder`.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=False,
        vocab_size=151936,
        num_layers=24,
        num_heads=14,
        num_kv_heads=2,
        embed_dim=896,
        intermediate_dim=4864,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen2_1_5b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Qwen2 1.5B model with LoRA enabled.

    The Qwen2 defaults are the same as in :func:`~torchtune.models.qwen2.qwen2_1_5b`,
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
        lora_dropout (float): dropout probability for the low-rank approximation. Default: 0.0
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Qwen2 1.5B model with LoRA applied

    Note:
        Qwen2 0.5B and Qwen2 1.5B model builders will enable `tie_word_embeddings` by default
        and returns an instance of `TransformerDecoder`.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=False,
        vocab_size=151936,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        embed_dim=1536,
        intermediate_dim=8960,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
