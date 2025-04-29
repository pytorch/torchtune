# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.qwen2._component_builders import lora_qwen2, qwen2
from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS, Qwen2_5Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json

"""
Model builders build specific instantiations using component builders. For example
the qwen2_5_7b model builder uses the qwen2 component builder to create the
Qwen2.5 7B model.
"""


def qwen2_5_0_5b() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 model (base or instruct) initialized w/ the default 0.5B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 0.5B model

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)
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
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_5_1_5b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 base model initialized w/ the default 1.5B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-1.5B

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 1.5B model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`).
    """
    return qwen2(
        vocab_size=151936,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        embed_dim=1536,
        intermediate_dim=8960,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_5_1_5b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 instruct model initialized w/ the default 1.5B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 1.5B instruct model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)
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
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_5_3b() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 model (base or instruct) initialized w/ the default 3B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 3B model

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)
    """
    return qwen2(
        vocab_size=151936,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        embed_dim=2048,
        intermediate_dim=11008,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        tie_word_embeddings=True,
    )


def qwen2_5_7b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 base model initialized w/ the default 7B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-7B

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 7B model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=28,
        num_heads=28,
        num_kv_heads=4,
        embed_dim=3584,
        intermediate_dim=18944,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
    )


def qwen2_5_7b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 instruct model initialized w/ the default 7B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 7B instruct model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
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
        norm_eps=1e-6,
        rope_base=1000000.0,
    )


def qwen2_5_14b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 base model initialized w/ the default 14B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-14B

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 14B model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
    )


def qwen2_5_14b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 instruct model initialized w/ the default 14B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-14B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 14B instruct model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
    )


def qwen2_5_32b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 base model initialized w/ the default 32B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-32B

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 32B model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=64,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=27648,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
    )


def qwen2_5_32b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 instruct model initialized w/ the default 32B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-32B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 32B instruct model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=64,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=27648,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
    )


def qwen2_5_72b_base() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 base model initialized w/ the default 72B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-72B

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 72B model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=29568,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
    )


def qwen2_5_72b_instruct() -> TransformerDecoder:
    """
    Builder for creating a Qwen2.5 instruct model initialized w/ the default 72B parameter values
    from https://huggingface.co/Qwen/Qwen2.5-72B-Instruct

    Returns:
        TransformerDecoder: Instantiation of Qwen2.5 72B instruct model

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return qwen2(
        vocab_size=152064,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=29568,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
    )


def qwen2_5_tokenizer(
    path: str,
    merges_file: str,
    special_tokens_path: Optional[str] = None,
    max_seq_len: Optional[int] = None,
    prompt_template: Optional[_TemplateType] = None,
    truncation_type: str = "right",
    **kwargs,
) -> Qwen2_5Tokenizer:
    """
    Tokenizer for Qwen2.5.

    Args:
        path (str): path to the vocab.json file.
        merges_file (str): path to the merges.txt file.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen2.5 special tokens.
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
        Qwen2_5Tokenizer: Instantiation of the Qwen2.5 tokenizer
    """
    special_tokens = (
        QWEN2_5_SPECIAL_TOKENS
        if special_tokens_path is None
        else parse_hf_tokenizer_json(special_tokens_path)
    )

    if prompt_template is not None:
        prompt_template = _get_prompt_template(prompt_template)

    return Qwen2_5Tokenizer(
        path=path,
        merges_file=merges_file,
        special_tokens=special_tokens,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
        truncation_type=truncation_type,
        **kwargs,
    )


def lora_qwen2_5_0_5b(
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
    Builder for creating a Qwen2.5 0.5B model (base or instruct) with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_0_5b`,
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
        TransformerDecoder: Instantiation of Qwen2.5 0.5B model with LoRA applied

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
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


def lora_qwen2_5_1_5b_base(
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
    Builder for creating a Qwen2.5 1.5B base model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_1_5b_base`,
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
        TransformerDecoder: Instantiation of Qwen2.5 1.5B model with LoRA applied

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=28,
        num_heads=12,
        num_kv_heads=2,
        embed_dim=1536,
        intermediate_dim=8960,
        max_seq_len=131072,
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


def lora_qwen2_5_1_5b_instruct(
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
    Builder for creating a Qwen2.5 1.5B instruct model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_1_5b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen2.5 1.5B model with LoRA applied

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
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


def lora_qwen2_5_3b(
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
    Builder for creating a Qwen2.5 3B model (base or instruct) with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_3b`,
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
        TransformerDecoder: Instantiation of Qwen2.5 3B model with LoRA applied

    Note:
        Qwen2.5 0.5B-3B model builders will enable ``tie_word_embeddings`` by default (see :func:`~torchtune.models.qwen2.qwen2`)
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=16,
        num_kv_heads=2,
        embed_dim=2048,
        intermediate_dim=11008,
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


def lora_qwen2_5_7b_base(
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
    Builder for creating a Qwen2.5 7B base model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_7b_base`,
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
        TransformerDecoder: Instantiation of Qwen2.5 7B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
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
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1000000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen2_5_7b_instruct(
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
    Builder for creating a Qwen2.5 7B instruct model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_7b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen2.5 7B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
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


def lora_qwen2_5_14b_base(
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
    Builder for creating a Qwen2.5 14B base model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_14b_base`,
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
        TransformerDecoder: Instantiation of Qwen2.5 14B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=13824,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen2_5_14b_instruct(
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
    Builder for creating a Qwen2.5 14B instruct model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_14b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen2.5 14B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=13824,
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


def lora_qwen2_5_32b_base(
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
    Builder for creating a Qwen2.5 32B base model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_32b_base`,
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
        TransformerDecoder: Instantiation of Qwen2.5 32B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=64,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=27648,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen2_5_32b_instruct(
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
    Builder for creating a Qwen2.5 32B instruct model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_32b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen2.5 32B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=64,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=5120,
        intermediate_dim=27648,
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


def lora_qwen2_5_72b_base(
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
    Builder for creating a Qwen2.5 72B base model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_72b_base`,
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
        TransformerDecoder: Instantiation of Qwen2.5 72B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=29568,
        max_seq_len=131072,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=1000000.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


def lora_qwen2_5_72b_instruct(
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
    Builder for creating a Qwen2.5 72B instruct model with LoRA enabled.

    The Qwen2.5 defaults are the same as in :func:`~torchtune.models.qwen2_5.qwen2_5_72b_instruct`,
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
        TransformerDecoder: Instantiation of Qwen2.5 72B model with LoRA applied

    Note:
        The base and instruct versions have slightly different architectures for all Qwen2.5 model sizes
        except 0.5B and 3B. Make sure to select the correct model builder for the weights.
    """
    return lora_qwen2(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=152064,
        num_layers=80,
        num_heads=64,
        num_kv_heads=8,
        embed_dim=8192,
        intermediate_dim=29568,
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
