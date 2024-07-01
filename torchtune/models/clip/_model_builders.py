# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtune.modules import VisionTransformer
from torchtune.models.clip._component_builders import clip

"""
Model builders build specific instantiations using component builders. For example
the clip_224_12 model builder uses the clip component builder to create the
clip with 12 layers and image size 224
"""

def clip_224_12() -> VisionTransformer:
    """
    Builder for creating a CLIP model initialized w/ the default 7B parameter values
    from https://arxiv.org/abs/2307.09288

    Returns:
        TransformerDecoder: Instantiation of Llama2 7B model
    """

    return clip(
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        tile_size=224,
        patch_size=16,
        max_num_tiles=4,
        mlp_ratio=4.0,
        act_layer=torch.nn.SiLU(),
        in_channels=3,
        attn_dropout=0.0,
        norm_eps=1e-5,
        cls_output_dim=512,
        output_cls_projection=True,
        indices_return_hidden=None,
    )
