# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.qwen3._component_builders import lora_qwen3_moe, qwen3_moe
from torchtune.models.qwen3._component_builders import lora_qwen3, qwen3
from torchtune.models.qwen3._tokenizer import QWEN3_SPECIAL_TOKENS, Qwen3Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json

"""
Model builders build specific instantiations using component builders. For example
the qwen3_8b_instruct model builder uses the qwen2 component builder to create the
Qwen3 8B instruct model.
"""


def qwen3_0_6b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 0.6B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-0.6B-Base

    Returns:
        TransformerDecoder: Instantiation of Qwen3 0.6B base model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_0_6b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 0.6B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-0.6B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 0.6B instruct model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_1_7b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 1.7B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-1.7B-Base

    Returns:
        TransformerDecoder: Instantiation of Qwen3 1.7B base model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        intermediate_dim=6144,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_1_7b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 1.7B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-1.7B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 1.7B instruct model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        intermediate_dim=6144,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_4b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 4B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-4B-Base

    Returns:
        TransformerDecoder: Instantiation of Qwen3 4B base model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_4b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 4B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-4B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 4B instruct model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_8b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 8B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-8B-Base

    Returns:
        TransformerDecoder: Instantiation of Qwen3 8B base model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_8b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 8B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-8B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 8B instruct model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_14b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 14B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-14B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 14B model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=40,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=17408,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_14b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 14B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-14B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen3 14B instruct model
    """
    return qwen3(
        vocab_size=151936,
        num_layers=40,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=17408,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_32b() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 32B model (instruct, no base variant) initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-32B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 32B instruct model (there's no base variant for the 32B)
    """
    return qwen3(
        vocab_size=151936,
        num_layers=64,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=25600,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_moe_30b_a3b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 30B A3B instruct model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-30B-A3B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 30B A3B instruct model
    """
    return qwen3_moe(
        vocab_size=151936,
        num_layers=48,
        num_heads=32,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_moe_30b_a3b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 30B A3B base model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-30B-A3B-Base

    Returns:
        TransformerDecoder: Instantiation of Qwen3 30B A3B base model
    """
    return qwen3_moe(
        vocab_size=151936,
        num_layers=48,
        num_heads=32,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_moe_235b_a22b() -> TransformerDecoder:
    """
    Builder for creating a Qwen3 235B A22B (instruct, no base variant) model initialized w/ the default parameter values
    from https://huggingface.co/Qwen/Qwen3-235B-A22B

    Returns:
        TransformerDecoder: Instantiation of Qwen3 235B A22B base model (there's no base variant for the 235B)
    """
    return qwen3_moe(
        vocab_size=151936,
        num_layers=94,
        num_heads=64,
        num_kv_heads=4,
        embed_dim=4096,
        intermediate_dim=12288,
        moe_intermediate_size=1536,
        num_experts=128,
        num_experts_per_tok=8,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
    )


def qwen3_tokenizer(
    path: str,
    merges_file: str,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
    **kwargs,
) -> Qwen3Tokenizer:
    """
    Tokenizer for Qwen3.

    Args:
        path (str): path to the vocab.json file.
        merges_file (str): path to the merges.txt file.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen3 special tokens.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.
            Default is None.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Returns:
        Qwen3Tokenizer: Instantiation of the Qwen3 tokenizer
    """
    special_tokens = (
        QWEN3_SPECIAL_TOKENS
        if special_tokens_path is None
        else parse_hf_tokenizer_json(special_tokens_path)
    )

    if prompt_template is not None:
        prompt_template = _get_prompt_template(prompt_template)

    return Qwen3Tokenizer(
        path=path,
        merges_file=merges_file,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
        truncation_type=truncation_type,
        **kwargs,
    )


def lora_qwen3_0_6b_base(
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
    Builder for creating a Qwen3 0.6B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_0_5b`,
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
        TransformerDecoder: Instantiation of Qwen3 0.6B model with LoRA applied

    Note:
        Qwen3 0.6B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_0_6b_instruct(
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
    Builder for creating a Qwen3 0.6B instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_1_5b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen3 0.6B model with LoRA applied

    Note:
        Qwen3 0.6B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)

    Note:
        The base and instruct versions have the exact same arch for all Qwen3 model sizes, except for `max_seq_len`. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=1024,
        intermediate_dim=3072,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_1_7b_base(
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
    Builder for creating a Qwen3 1.7B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_1_5b_base`,
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
        TransformerDecoder: Instantiation of Qwen3 1.7B model with LoRA applied

    Note:
        Qwen3 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        intermediate_dim=6144,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_1_7b_instruct(
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
    Builder for creating a Qwen3 1.7B instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_1_5b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen3 1.7B model with LoRA applied

    Note:
        Qwen3 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        embed_dim=2048,
        intermediate_dim=6144,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_4b_base(
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
    Builder for creating a Qwen3 4B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_3b`,
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
        TransformerDecoder: Instantiation of Qwen3 4B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_4b_instruct(
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
    Builder for creating a Qwen3 4B instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_3b`,
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
        TransformerDecoder: Instantiation of Qwen3 4B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_8b_base(
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
    Builder for creating a Qwen3 8B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_7b_base`,
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
        TransformerDecoder: Instantiation of Qwen3 8B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_8b_instruct(
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
    Builder for creating a Qwen3 8B instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_7b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen3 8B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=12288,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_14b_base(
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
    Builder for creating a Qwen3 14B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_14b_base`,
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
        TransformerDecoder: Instantiation of Qwen3 14B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=40,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=17408,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_14b_instruct(
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
    Builder for creating a Qwen3 14B instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_14b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen3 14B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=40,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=17408,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_32b(
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
    Builder for creating a Qwen3 32B model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_32b_base`,
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
        TransformerDecoder: Instantiation of Qwen3 32B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=64,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=25600,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_moe_30b_a3b_instruct(
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
    Builder for creating a Qwen3 MoE 30B A3B instruct model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen3.qwen3_moe_30b_a3b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen3 MoE 30B A3B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3_moe(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=48,
        num_heads=32,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_moe_30b_a3b_base(
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
    Builder for creating a Qwen3 MoE 30B A3B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen3.qwen3_moe_30b_a3b_base`,
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
        TransformerDecoder: Instantiation of Qwen3 MoE 30B A3B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3_moe(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=48,
        num_heads=32,
        num_kv_heads=4,
        embed_dim=2048,
        intermediate_dim=6144,
        moe_intermediate_size=768,
        num_experts=128,
        num_experts_per_tok=8,
        max_seq_len=32768,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen3_moe_235b_a22b(
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
    Builder for creating a Qwen3 MoE 235B A22B base model with LoRA enabled.

    The Qwen3 defaults are the same as in :func:`~torchtune.models.qwen3.qwen3_moe_235b_a22b`,
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
        TransformerDecoder: Instantiation of Qwen3 MoE 30B A3B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen3 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen3_moe(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=94,
        num_heads=64,
        num_kv_heads=4,
        embed_dim=4096,
        intermediate_dim=12288,
        moe_intermediate_size=1536,
        num_experts=128,
        num_experts_per_tok=8,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
