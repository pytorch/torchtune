# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional

import torch
from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType
from torchtune.models.llama3_2_vision._component_builders import (  # noqa
    llama3_2_vision_decoder,
    llama3_2_vision_encoder,
    lora_llama3_2_vision_decoder,
    lora_llama3_2_vision_encoder,
    LoRATrainable,
)
from torchtune.models.llama3_2_vision._encoder import Llama3VisionEncoder
from torchtune.models.llama3_2_vision._transform import Llama3VisionTransform
from torchtune.modules.model_fusion import DeepFusionModel
from torchtune.modules.peft import LORA_ATTN_MODULES
from torchtune.modules.tokenizers import parse_hf_tokenizer_json


def llama3_2_vision_11b(
    decoder_trainable: bool = False,
    encoder_trainable: bool = True,
    fusion_trainable: bool = True,
    image_size: int = 560,
) -> DeepFusionModel:
    """Llama 3.2 Vision 11B model

    Args:
        decoder_trainable (bool): Whether to make decoder params trainable. Default is False.
        encoder_trainable (bool): Whether to make encoder params trainable. Default is True.
        fusion_trainable (bool): Whether to make fusion params trainable. Default is True.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 560 for Instruct weights, use 448 for pre-trained.

    Returns:
        DeepFusionModel: Instantiation of the Llama 3.2 Vision 11B model
    """
    encoder = llama3_2_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1280,
        clip_num_layers=32,
        clip_hidden_states=[3, 7, 15, 23, 30],
        decoder_embed_dim=4096,
        num_layers_projection=8,
        tile_size=image_size,
        max_num_tiles=4,
        in_channels=3,
    )
    decoder = llama3_2_vision_decoder(
        vocab_size=128_256,
        num_layers=32,
        fusion_interval=4,
        num_special_tokens=8,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131_072,
        encoder_max_seq_len=128_080,  # 20*6404
        rope_base=500000.0,
        intermediate_dim=14336,
    )
    return DeepFusionModel(
        encoder=encoder,
        decoder=decoder,
        encoder_trainable=encoder_trainable,
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )


def llama3_2_vision_transform(
    path: str,
    max_seq_len: int = 8192,
    image_size: int = 560,
    special_tokens_path: Optional[str] = None,
    prompt_template: Optional[_TemplateType] = None,
) -> Llama3VisionTransform:
    """
    Data Transforms (including Tokenizer) for Llama3 Vision.

    Args:
        path (str): path to the tokenizer
        max_seq_len (int): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 560 for Instruct weights, use 448 for pre-trained.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        Llama3VisionTransform: Instantiation of the Llama 3.2 vision transform
    """
    special_tokens = (
        parse_hf_tokenizer_json(special_tokens_path)
        if special_tokens_path is not None
        else None
    )
    template = (
        _get_prompt_template(prompt_template) if prompt_template is not None else None
    )
    return Llama3VisionTransform(
        path=path,
        special_tokens=special_tokens,
        tile_size=image_size,
        patch_size=14,
        max_num_tiles=4,
        max_seq_len=max_seq_len,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        prompt_template=template,
    )


def lora_llama3_2_vision_11b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    decoder_trainable: str = "frozen",
    encoder_trainable: str = "lora",
    fusion_trainable: str = "lora",
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    image_size: int = 560,
) -> DeepFusionModel:
    """
    Return a version of Llama3.2 vision (an instance of :func:`~torchtune.modules.model_fusion.DeepFusionModel`)
    with LoRA applied based on the passed in configuration.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        decoder_trainable (str): Option to set decoder params as fully trainble (full), lora trainable (lora),
            or frozen (frozen). The default is "frozen".
        encoder_trainable (str): Option to set encoder params as fully trainble (full), lora trainable (lora),
            or frozen (frozen). The default is "lora".
        fusion_trainable (str): Option to set fusion params as fully trainble (full), lora trainable (lora),
            or frozen (frozen). The default is "lora".
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 560 for Instruct weights, use 448 for pre-trained.

    Returns:
        DeepFusionModel: Instantiation of Llama3.2 vision model with LoRA applied to
        a subset of the attention projections in each layer.

    """
    decoder_type = LoRATrainable(decoder_trainable.lower())
    encoder_type = LoRATrainable(encoder_trainable.lower())
    fusion_type = LoRATrainable(fusion_trainable.lower())
    encoder = lora_llama3_2_vision_encoder(
        encoder_lora=encoder_type == LoRATrainable.LORA,
        fusion_lora=fusion_type == LoRATrainable.LORA,
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1280,
        clip_num_layers=32,
        clip_hidden_states=[3, 7, 15, 23, 30],
        decoder_embed_dim=4096,
        num_layers_projection=8,
        tile_size=image_size,
        max_num_tiles=4,
        in_channels=3,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
    decoder = lora_llama3_2_vision_decoder(
        decoder_lora=decoder_type == LoRATrainable.LORA,
        fusion_lora=fusion_type == LoRATrainable.LORA,
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=32,
        fusion_interval=4,
        num_special_tokens=8,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=131_072,
        encoder_max_seq_len=128_080,  # 20*6404
        rope_base=500000.0,
        intermediate_dim=14336,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
    return DeepFusionModel(
        encoder=encoder,
        decoder=decoder,
        encoder_trainable=encoder_type != LoRATrainable.FROZEN,
        decoder_trainable=decoder_type != LoRATrainable.FROZEN,
        fusion_trainable=fusion_type != LoRATrainable.FROZEN,
    )


qlora_llama3_2_vision_11b = partial(lora_llama3_2_vision_11b, quantize_base=True)

qlora_llama3_2_vision_11b.__doc__ = """
Builder for creating a Llama3.2 vision 11B model with QLoRA enabled. Base model weights in linear layers
that LoRA is applied to are quantized per the QLoRA paper: https://arxiv.org/abs/2305.14314.
Please see `lora_llama3_2_vision_11b` for full API arguments.
"""
