# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

from torchtune.data._prompt_templates import _TemplateType

from torchtune.models.llama4 import llama4_decoder, llama4_vision_encoder
from torchtune.models.llama4._tokenizer import LLAMA4_SPECIAL_TOKENS
from torchtune.models.llama4._transform import Llama4Transform

from torchtune.modules.model_fusion import EarlyFusionModel

"""
Model builders build specific instantiations using component builders. For example
the llama4 model builder uses the MoE component builder to create the
Llama4 MoE model.
"""


def llama4_scout_17b_16e(
    decoder_trainable: bool = True,
    vision_encoder_trainable: bool = False,
    fusion_trainable: bool = True,
    image_size: int = 336,
) -> EarlyFusionModel:
    """
    Builder for creating an instance of the Llama4 Scout 17Bx16E model

    Args:
        decoder_trainable (bool): Whether to make decoder params trainable. Default is True.
        encoder_trainable (bool): Whether to make encoder params trainable. Default is False.
        fusion_trainable (bool): Whether to make fusion params trainable. Default is True.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 336.

    Returns:
        EarlyFusionModel: Instantiation of a 17Bx16E Llama4 MoE model with encoders.
    """
    decoder_embed_dim = 5120

    vision_encoder = llama4_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1408,
        clip_num_layers=34,
        decoder_embed_dim=decoder_embed_dim,
        projection_embed_dim=4096,
        tile_size=image_size,
        max_num_tiles=16,
        in_channels=3,
    )

    decoder = llama4_decoder(
        vocab_size=202_048,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=decoder_embed_dim,
        hidden_dim=8192,
        max_seq_len=10485760,
        attn_dropout=0.0,
        rope_base=500_000,
        norm_eps=1e-5,
        num_experts=16,
        use_shared_expert=True,
        skip_rope_interval=4,
        attention_chunk_size=8192,
        use_scaled_rope=True,
    )
    return EarlyFusionModel(
        decoder,
        encoders={"vision": vision_encoder},
        encoder_tokens={
            "vision": LLAMA4_SPECIAL_TOKENS["<|patch|>"],
        },
        encoders_trainable={
            "vision": vision_encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )


def llama4_maverick_17b_128e(
    decoder_trainable: bool = True,
    vision_encoder_trainable: bool = False,
    fusion_trainable: bool = True,
    image_size: int = 336,
) -> EarlyFusionModel:
    """
    Builder for creating an instance of the Llama4 Maverick 17Bx128E model

    Args:
        decoder_trainable (bool): Whether to make decoder params trainable. Default is True.
        encoder_trainable (bool): Whether to make encoder params trainable. Default is False.
        fusion_trainable (bool): Whether to make fusion params trainable. Default is True.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 336.

    Returns:
        EarlyFusionModel: Instantiation of a 17Bx128E Llama4 MoE model with encoders.
    """
    decoder_embed_dim = 5120

    vision_encoder = llama4_vision_encoder(
        patch_size=14,
        num_heads=16,
        clip_embed_dim=1408,
        clip_num_layers=34,
        decoder_embed_dim=decoder_embed_dim,
        projection_embed_dim=4096,
        tile_size=image_size,
        max_num_tiles=16,
        in_channels=3,
    )

    decoder = llama4_decoder(
        vocab_size=202_048,
        num_layers=48,
        num_heads=40,
        num_kv_heads=8,
        embed_dim=decoder_embed_dim,
        hidden_dim=8192,
        max_seq_len=1048576,
        attn_dropout=0.0,
        rope_base=500_000,
        norm_eps=1e-5,
        num_experts=128,
        use_shared_expert=True,
        use_qk_norm=False,
        moe_every_n_layers=2,
        mlp_hidden_dim=16384,
        skip_rope_interval=4,
        attention_chunk_size=8192,
    )
    return EarlyFusionModel(
        decoder,
        encoders={"vision": vision_encoder},
        encoder_tokens={
            "vision": LLAMA4_SPECIAL_TOKENS["<|patch|>"],
        },
        encoders_trainable={
            "vision": vision_encoder_trainable,
        },
        decoder_trainable=decoder_trainable,
        fusion_trainable=fusion_trainable,
    )


def llama4_transform(
    path: str,
    max_seq_len: int = 8192,
    image_size: int = 336,
    max_num_tiles: int = 16,
    special_tokens_path: Optional[str] = None,
    prompt_template: Optional[_TemplateType] = None,
) -> Llama4Transform:
    """
    Data transform (including tokenizer) for Llama4.

    Args:
        path (str): path to the tokenizer
        max_seq_len (int): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated.
        image_size (int): Base image size that images will be tiled and resized to.
            Default is 336.
        max_num_tiles (int): Maximum number of tiles to use for each image. Default is 16.
        special_tokens_path (Optional[str]): Path to ``tokenizer.json`` from Hugging Face
            model files that contains all registered special tokens, or a local json file
            structured similarly. Default is None to use the canonical Llama3 special tokens.
        prompt_template (Optional[_TemplateType]): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        Llama4Transform: Instantiation of the Llama 4 transform
    """
    return Llama4Transform(
        path=path,
        special_tokens_path=special_tokens_path,
        tile_size=image_size,
        patch_size=14,
        max_num_tiles=max_num_tiles,
        max_seq_len=max_seq_len,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        prompt_template=prompt_template,
    )
