# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
import torch.nn as nn

from torchtune.data._prompt_templates import _TemplateType

from torchtune.models.qwen2_5_vision._component_builders import (
    qwen2_5_vl_text_decoder,
    qwen2_5_vision_encoder,
)

from torchtune.models.qwen2_5_vision._transform import Qwen2_5_VLTransform
from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS
from torchtune.models.qwen2_5_vision._mrope_early_fusion import Qwen25VLEarlyFusionModel

"""
Model builders build specific instantiations using component builders. 
"""

def qwen2_5_vl_3b(
    *,
    decoder_trainable: bool = True,
    encoder_trainable: bool = True,
    fusion_trainable: bool = False,
    image_size: int = 336,
) -> Qwen25VLEarlyFusionModel:
    """
    Builder for creating a Qwen2.5-VL 3B base model with vision capabilities.
    
    Args:
        decoder_trainable (bool): Whether the language model decoder should be trainable. Default: False
        encoder_trainable (bool): Whether the vision encoder should be trainable. Default: False
        fusion_trainable (bool): Whether the fusion layers should be trainable. Default: False
        image_size (int): Input image size for the vision encoder. Default: 336
    """
    
    encoder = qwen2_5_vision_encoder(
        embed_dim=1280,
        num_layers=32,
        activation=nn.SiLU(),
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        out_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        full_att_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
    )

    decoder = qwen2_5_vl_text_decoder(
        vocab_size=152064, 
        num_layers=36,
        num_kv_heads=2, 
        embed_dim=3584,
        intermediate_dim=4864,
        max_seq_len=32768,
        attn_dropout=0.0,
        rope_base=1000000.0,
        norm_eps=1e-6,
        mrope_section=[16, 24, 24],
        tie_word_embeddings=True,
    )

    return Qwen25VLEarlyFusionModel(
        decoder=decoder,
        encoders={"image": encoder},
        encoder_tokens={
            "image": QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],
        },
        image_token_id=QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],
        vision_start_token_id=QWEN2_5_SPECIAL_TOKENS["<|vision_start|>"],
        spatial_merge_size=2,
        tokens_per_second=2,  
        encoders_trainable={
            "image": encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )

def qwen2_5_vl_7b(
    *,
    decoder_trainable: bool = True,
    encoder_trainable: bool = True,
    fusion_trainable: bool = False,
    image_size: int = 336,
) -> Qwen25VLEarlyFusionModel:
    """
    Builder for creating a Qwen2.5-VL 7B base model with vision capabilities.
    
    Args:
        decoder_trainable (bool): Whether the language model decoder should be trainable. Default: False
        encoder_trainable (bool): Whether the vision encoder should be trainable. Default: False
        fusion_trainable (bool): Whether the fusion layers should be trainable. Default: False
        image_size (int): Input image size for the vision encoder. Default: 336
        
    Returns:
        Qwen25VLEarlyFusionModel: Qwen2.5-VL 7B model instance
    """

    encoder = qwen2_5_vision_encoder(
        embed_dim=1280,
        num_layers=32,
        activation=nn.SiLU(),
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        out_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        full_att_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
    )

    decoder = qwen2_5_vl_text_decoder(
        vocab_size=152064, 
        num_layers=28,
        num_kv_heads=4, 
        embed_dim=3584,
        intermediate_dim=18944,
        max_seq_len=32768,
        attn_dropout=0.0,
        rope_base=1000000.0,
        norm_eps=1e-6,
        mrope_section=[16, 24, 24],
        tie_word_embeddings=False,
    )

    return Qwen25VLEarlyFusionModel(
        decoder=decoder,
        encoders={"image": encoder},  
        encoder_tokens={
            "image": QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],    
        },
        image_token_id=QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],
        vision_start_token_id=QWEN2_5_SPECIAL_TOKENS["<|vision_start|>"],
        spatial_merge_size=2,
        tokens_per_second=2,  
        encoders_trainable={
            "image": encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )

def qwen2_5_vl_72b(
    *,
    decoder_trainable: bool = True,
    encoder_trainable: bool = True,
    fusion_trainable: bool = False,
    image_size: int = 336,
) -> Qwen25VLEarlyFusionModel:
    """
    Builder for creating a Qwen2.5-VL 72B base model with vision capabilities.
    
    Args:
        decoder_trainable (bool): Whether the language model decoder should be trainable. Default: False
        encoder_trainable (bool): Whether the vision encoder should be trainable. Default: False
        fusion_trainable (bool): Whether the fusion layers should be trainable. Default: False
        image_size (int): Input image size for the vision encoder. Default: 336
        
    Returns:
        Qwen25VLEarlyFusionModel: Qwen2.5-VL 72B model instance
    """

    encoder = qwen2_5_vision_encoder(
        embed_dim=1280,
        num_layers=32,
        activation=nn.SiLU(),
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        out_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        window_size=112,
        full_att_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
    )

    decoder = qwen2_5_vl_text_decoder(
        vocab_size=152064, 
        num_layers=80,
        num_kv_heads=8, 
        embed_dim=3584,
        intermediate_dim=29568,
        max_seq_len=32768,
        attn_dropout=0.0,
        rope_base=1000000.0,
        norm_eps=1e-6,
        mrope_section=[16, 24, 24],
        tie_word_embeddings=False,
    )

    return Qwen25VLEarlyFusionModel(
        decoder=decoder,
        encoders={"image": encoder},
        encoder_tokens={
            "image": QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],
        },
        image_token_id=QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],
        vision_start_token_id=QWEN2_5_SPECIAL_TOKENS["<|vision_start|>"],
        spatial_merge_size=2,
        tokens_per_second=2,  
        encoders_trainable={
            "image": encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )

def qwen2_5_vl_transform(
    path: str,
    merges_file: str,
    max_seq_len: Optional[int] = None,
    patch_size: Optional[int] = None, 
    prompt_template: Optional[_TemplateType] = None,
) -> Qwen2_5_VLTransform:
    """
    Data transform (including tokenizer) for Qwen2.5-VL.

    Args:
        path (str): path to the vocab.json file
        merges_file (str): path to the merges.txt file
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated.
        patch_size (Optional[int]): Size of the patches to divide the image into.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        Qwen2_5_VLTransform: Instantiation of the Qwen2.5-VL transform
    """
    return Qwen2_5_VLTransform(
        path=path,
        merges_file=merges_file,
        patch_size=patch_size,
        max_seq_len=max_seq_len, 
        prompt_template=prompt_template, 
    )