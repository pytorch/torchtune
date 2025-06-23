# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Optional

from torchtune.data._prompt_templates import _get_prompt_template, _TemplateType

from torchtune.models.qwen2_5._model_builders import qwen2_5_7b_base, qwen2_5_7b_instruct
from torchtune.models.qwen2_5._tokenizer import QWEN2_5_SPECIAL_TOKENS, Qwen2_5Tokenizer
from torchtune.models.qwen2_5_vision._encoder import Qwen2_5_VisionTransformer
from torchtune.modules import TransformerDecoder
from torchtune.modules.model_fusion import EarlyFusionModel
from torchtune.modules.transforms.tokenizers import parse_hf_tokenizer_json

"""
Model builders build specific instantiations using component builders. For example
the qwen2_5_vl_7b_base model builder uses the qwen2_5_7b_base component builder to create the
Qwen2.5-VL 7B model with vision capabilities.
"""


def qwen2_5_vl_7b_base(
    *,
    decoder_trainable: bool = True,
    encoder_trainable: bool = False,
    fusion_trainable: bool = True,
    image_size: int = 336,
) -> EarlyFusionModel:
    """
    Builder for creating a Qwen2.5-VL 7B base model with vision capabilities.
    
    This combines:
    - Qwen2.5 7B base language model as the decoder backbone
    - Vision transformer encoder for processing images
    - Early fusion architecture for multimodal understanding
    
    Args:
        decoder_trainable (bool): Whether the language model decoder should be trainable. Default: True
        encoder_trainable (bool): Whether the vision encoder should be trainable. Default: False
        fusion_trainable (bool): Whether the fusion layers should be trainable. Default: True
        image_size (int): Input image size for the vision encoder. Default: 336
        
    Returns:
        EarlyFusionModel: Qwen2.5-VL 7B model instance
    """

    decoder = qwen2_5_7b_base()

    # TODO: FINALIZE VISION ENCODER ARGS - This will be completed by the vision team
    encoder = Qwen2_5_VisionTransformer(
        patch_size=14,
        tile_size=image_size,
        num_layers=32,
        embed_dim=1280,
        layer=...,  # To be completed by vision encoder implementation
        token_pos_embedding=...,  # To be completed by vision encoder implementation
        pre_tile_pos_embed=None,
        post_tile_pos_embed=None,
        cls_projection=None,
        out_indices=[7, 15, 23, 31],
        in_channels=3,
        append_cls_token=False,
    )

    return EarlyFusionModel(
        decoder=decoder,
        encoder={"vision": encoder},
        encoder_tokens={
            "vision": QWEN2_5_SPECIAL_TOKENS["<|vision_pad|>"],  # Use the proper vision token
        },
        encoders_trainable={
            "vision": encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )