# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn

from torchtune.models.qwen2_5_vision._component_builders import (
    qwen2_5_vl_text_decoder,
    qwen2_5_vision_encoder,
)

from torchtune.models.qwen2_5_vision._transform import Qwen2_5_VLTransform
from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS, Qwen2_5Tokenizer
from torchtune.models.qwen2_5_vision._encoder import Qwen2_5_VisionTransformer
from torchtune.models.qwen2_5_vision._mrope_early_fusion import Qwen25VLEarlyFusionModel
from torchtune.utils import torch_version_ge

"""
Model builders build specific instantiations using component builders. For example
the qwen2_5_vl_7b_base model builder uses the qwen2_5_7b_base component builder to create the
Qwen2.5-VL 7B model with vision capabilities.
"""


def qwen2_5_vl_7b(
    *,
    decoder_trainable: bool = False,
    encoder_trainable: bool = False,
    fusion_trainable: bool = False,
    image_size: int = 336,
) -> Qwen25VLEarlyFusionModel:
    """
    Builder for creating a Qwen2.5-VL 7B base model with vision capabilities.
    
    This combines:
    - Qwen2.5 7B base language model as the decoder backbone
    - Vision transformer encoder for processing images and videos
    - Early fusion architecture for multimodal understanding
    
    Args:
        decoder_trainable (bool): Whether the language model decoder should be trainable. Default: False
        encoder_trainable (bool): Whether the vision encoder should be trainable. Default: False
        fusion_trainable (bool): Whether the fusion layers should be trainable. Default: False
        image_size (int): Input image size for the vision encoder. Default: 336
        
    Returns:
        Qwen25VLEarlyFusionModel: Qwen2.5-VL 7B model instance
    """
    # TODO: add version check; copied from llama4
    # assert torch_version_ge("2.8"), "Qwen2.5-VL requires Pytorch 2.8 or higher"

    decoder = qwen2_5_vl_text_decoder(
        vocab_size=152064, # TODO: check if this value from hf/config.json is correct; paper says 151646
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

    # Single encoder handles both images and videos
    encoder = qwen2_5_vision_encoder(
        embed_dim=1280,
        num_layers=32,
        activation=nn.SiLU,
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        out_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        # spatial_patch_size=14,
        window_size=112,
        full_att_block_indexes=[7, 15, 23, 31],
        temporal_patch_size=2,
        # tokens_per_second=2 # NOTE: needed for get_rope_index
    )

    return Qwen25VLEarlyFusionModel(
        decoder=decoder,
        encoders={"image": encoder, "video": encoder},  # Same encoder for both
        encoder_tokens={
            "image": QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],    # 151655
            "video": QWEN2_5_SPECIAL_TOKENS["<|video_pad|>"],    # 151656
        },
        # Use the correct special token IDs
        image_token_id=QWEN2_5_SPECIAL_TOKENS["<|image_pad|>"],
        video_token_id=QWEN2_5_SPECIAL_TOKENS["<|video_pad|>"],
        vision_start_token_id=QWEN2_5_SPECIAL_TOKENS["<|vision_start|>"],
        spatial_merge_size=2,
        tokens_per_second=2,  
        encoders_trainable={
            "image": encoder_trainable,
            "video": encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )