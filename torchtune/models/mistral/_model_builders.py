# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.models.mistral._component_builders import (
    mistral,
    lora_mistral,
    mistral_classifier,
    lora_mistral_classifier,
)
from torchtune.data._prompt_templates import _TemplateType
from torchtune.data._prompt_templates import _get_prompt_template

from torchtune.modules import TransformerDecoder
from torchtune.models.mistral._tokenizer import MistralTokenizer
from torchtune.modules.peft import LORA_ATTN_MODULES
from functools import partial


"""
Model builders build specific instantiations using component builders. For example
the ``mistral_7b`` model builder uses the ``mistral`` component builder.
"""


def mistral_7b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b parameter values
    from https://mistral.ai/news/announcing-mistral-7b/


    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model
    """
    return mistral(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def mistral_tokenizer(path: str, max_seq_len: Optional[int] = None, prompt_template: Optional[_TemplateType] = "torchtune.models.mistral.MistralChatTemplate") -> MistralTokenizer:
    """
    Tokenizer for Mistral models.

    Args:
        path (str): path to the tokenizer
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags. Default is :class:`~torchtune.models.mistral.MistralChatTemplate`.

    Returns:
        MistralTokenizer: Instantiation of the Mistral tokenizer
    """
    return MistralTokenizer(path=path, max_seq_len=max_seq_len, prompt_template=_get_prompt_template(prompt_template) if prompt_template is not None else None)


def lora_mistral_7b(
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
    Builder for creating a Mistral 7B model with LoRA enabled.

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
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model with LoRA applied
    """
    return lora_mistral(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_mistral_7b = partial(lora_mistral_7b, quantize_base=True)

qlora_mistral_7b.__doc__ = """
Builder for creating a Mistral model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_mistral_7b` for full API arguments.
"""


def mistral_reward_7b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b
    parameter values from:
    https://huggingface.co/Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback
    where the output layer is a classification layer projecting to a single class for reward modelling.

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B classifier model
    """
    return mistral_classifier(
        num_classes=1,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def lora_mistral_reward_7b(
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
    Builder for creating a Mistral reward 7B model with LoRA enabled.

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
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model with LoRA applied
    """
    return lora_mistral_classifier(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        num_classes=1,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=10_000,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )


qlora_mistral_reward_7b = partial(lora_mistral_reward_7b, quantize_base=True)

qlora_mistral_reward_7b.__doc__ = """
Builder for creating a Mistral reward 7B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_mistral_reward_7b` for full API arguments.
"""
