# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional
from functools import partial

from torchtune.models.qwen2._component_builders import qwen2, lora_qwen2
from torchtune.models.qwen2._tokenizer import Qwen2Tokenizer

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.tokenizers import parse_hf_tokenizer_json

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


def qwen2_tokenizer(
        vocab_file: str, merges_file: str, special_tokens_path: Optional[str] = None,
        **kwargs,
) -> Qwen2Tokenizer:
    """
    Tokenizer for Qwen2.

    Args:
        vocab_file (str): path to the vocab file.
        merges_file (str): path to the merges file.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Qwen2 special tokens.

    Returns:
        Llama3Tokenizer: Instantiation of the Qwen2 tokenizer
    """
    special_tokens = parse_hf_tokenizer_json(special_tokens_path) if special_tokens_path is not None else None
    return Qwen2Tokenizer(vocab_file=vocab_file, merges_file=merges_file, special_tokens=special_tokens, **kwargs)


def lora_qwen2_7b(
        lora_attn_modules: List[LORA_ATTN_MODULES],
        apply_lora_to_mlp: bool = False,
        apply_lora_to_output: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.05,
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
        quantize_base=quantize_base,
    )


qlora_qwen2_7b = partial(lora_qwen2_7b, quantize_base=True)

qlora_qwen2_7b.__doc__ = """
Builder for creating a Qwen2 7B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_qwen2_7b` for full API arguments.
"""
