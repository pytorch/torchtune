# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

_FROM_META = {
    "text_model.tok_embeddings.weight": "decoder.tok_embeddings.weight",
    "text_model.learnable_embedding.weight": "decoder.tok_embeddings.fusion_embedding.weight",
    "text_model.norm.weight": "decoder.norm.scale",
    "text_model.output.weight": "decoder.output.weight",
    "text_model.layers.{}.attention_norm.weight": "decoder.layers.{}.sa_norm.scale",
    "text_model.layers.{}.attention.wq.weight": "decoder.layers.{}.attn.q_proj.weight",
    "text_model.layers.{}.attention.wk.weight": "decoder.layers.{}.attn.k_proj.weight",
    "text_model.layers.{}.attention.wv.weight": "decoder.layers.{}.attn.v_proj.weight",
    "text_model.layers.{}.attention.wo.weight": "decoder.layers.{}.attn.output_proj.weight",
    "text_model.layers.{}.ffn_norm.weight": "decoder.layers.{}.mlp_norm.scale",
    "text_model.layers.{}.feed_forward.w1.weight": "decoder.layers.{}.mlp.w1.weight",
    "text_model.layers.{}.feed_forward.w3.weight": "decoder.layers.{}.mlp.w3.weight",
    "text_model.layers.{}.feed_forward.w2.weight": "decoder.layers.{}.mlp.w2.weight",
    "text_model.cross_attention_layers.{}.gate_attn": "decoder.layers.{}.fusion_layer.ca_scale.scale",
    "text_model.cross_attention_layers.{}.gate_ffwd": "decoder.layers.{}.fusion_layer.mlp_scale.scale",
    "text_model.cross_attention_layers.{}.attention_norm.weight": "decoder.layers.{}.fusion_layer.ca_norm.scale",
    "text_model.cross_attention_layers.{}.ffn_norm.weight": "decoder.layers.{}.fusion_layer.mlp_norm.scale",
    "text_model.cross_attention_layers.{}.attention.wq.weight": "decoder.layers.{}.fusion_layer.attn.q_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wk.weight": "decoder.layers.{}.fusion_layer.attn.k_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wv.weight": "decoder.layers.{}.fusion_layer.attn.v_proj.weight",
    "text_model.cross_attention_layers.{}.attention.wo.weight": "decoder.layers.{}.fusion_layer.attn.output_proj.weight",
    "text_model.cross_attention_layers.{}.attention.q_norm.weight": "decoder.layers.{}.fusion_layer.attn.q_norm.scale",
    "text_model.cross_attention_layers.{}.attention.k_norm.weight": "decoder.layers.{}.fusion_layer.attn.k_norm.scale",
    "text_model.cross_attention_layers.{}.feed_forward.w1.weight": "decoder.layers.{}.fusion_layer.mlp.w1.weight",
    "text_model.cross_attention_layers.{}.feed_forward.w3.weight": "decoder.layers.{}.fusion_layer.mlp.w3.weight",
    "text_model.cross_attention_layers.{}.feed_forward.w2.weight": "decoder.layers.{}.fusion_layer.mlp.w2.weight",
    "vision_model.vision_encoder.positional_embedding": "encoder.clip.token_pos_embedding.local_token_positional_embedding",
    "vision_model.vision_encoder.gated_positional_embedding": "encoder.clip.token_pos_embedding.global_token_positional_embedding",
    "vision_model.vision_encoder.gated_positional_embedding_gate": "encoder.clip.token_pos_embedding.gate",
    "vision_model.vision_encoder.ln_pre.weight": "encoder.clip.ln_pre.weight",
    "vision_model.vision_encoder.ln_pre.bias": "encoder.clip.ln_pre.bias",
    "vision_model.vision_encoder.ln_post.weight": "encoder.clip.ln_post.weight",
    "vision_model.vision_encoder.ln_post.bias": "encoder.clip.ln_post.bias",
    "vision_model.vision_encoder.pre_tile_pos_embed.embedding": "encoder.clip.pre_tile_pos_embed.embedding",
    "vision_model.vision_encoder.pre_tile_pos_embed.gate": "encoder.clip.pre_tile_pos_embed.gate",
    "vision_model.vision_encoder.post_tile_pos_embed.embedding": "encoder.clip.post_tile_pos_embed.embedding",
    "vision_model.vision_encoder.post_tile_pos_embed.gate": "encoder.clip.post_tile_pos_embed.gate",
    "vision_model.vision_encoder.class_embedding": "encoder.clip.cls_token_embedding.weight",
    "vision_model.vision_encoder.conv1._linear.weight": "encoder.clip.conv.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wq.weight": "encoder.clip.layers.{}.attn.q_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wk.weight": "encoder.clip.layers.{}.attn.k_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wv.weight": "encoder.clip.layers.{}.attn.v_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.attn.wo.weight": "encoder.clip.layers.{}.attn.output_proj.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_fc.weight": "encoder.clip.layers.{}.mlp.w1.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_fc.bias": "encoder.clip.layers.{}.mlp.w1.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_proj.weight": "encoder.clip.layers.{}.mlp.w2.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.mlp.c_proj.bias": "encoder.clip.layers.{}.mlp.w2.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_1.weight": "encoder.clip.layers.{}.sa_norm.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_1.bias": "encoder.clip.layers.{}.sa_norm.bias",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_2.weight": "encoder.clip.layers.{}.mlp_norm.weight",
    "vision_model.vision_encoder.transformer.resblocks.{}.ln_2.bias": "encoder.clip.layers.{}.mlp_norm.bias",
    "vision_model.vision_projection.weight": "encoder.projection.output.weight",
    "vision_model.vision_projection.bias": "encoder.projection.output.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wq.weight": "encoder.projection.layers.{}.attn.q_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wk.weight": "encoder.projection.layers.{}.attn.k_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wv.weight": "encoder.projection.layers.{}.attn.v_proj.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.attn.wo.weight": "encoder.projection.layers.{}.attn.output_proj.weight",  # noqa
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.weight": "encoder.projection.layers.{}.mlp.w1.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_fc.bias": "encoder.projection.layers.{}.mlp.w1.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.weight": "encoder.projection.layers.{}.mlp.w2.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.mlp.c_proj.bias": "encoder.projection.layers.{}.mlp.w2.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_1.weight": "encoder.projection.layers.{}.sa_norm.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_1.bias": "encoder.projection.layers.{}.sa_norm.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_2.weight": "encoder.projection.layers.{}.mlp_norm.weight",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.ln_2.bias": "encoder.projection.layers.{}.mlp_norm.bias",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.gate_attn": "encoder.projection.layers.{}.sa_scale.scale",
    "vision_model.vision_encoder.global_transformer.resblocks.{}.gate_ffn": "encoder.projection.layers.{}.mlp_scale.scale",
}


def _layer_num(key: str):
    """Get layer number from key or return None"""
    layer_num = [int(k) for k in key.split(".") if k.isdigit()]
    if len(layer_num) > 1:
        raise ValueError("More than one number in key, ambiguous input")
    elif len(layer_num) == 1:
        return int(layer_num[0])
    else:
        return None


def llama3_vision_meta_to_tune(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convertor from Meta state dict to torchtune state dict. This handles:
    - Updateing the cross attention layer numbers
    - reshaping the convolution weights
    - skip loading the rope embeddings
    """
    converted_state_dict = {}

    # Calculate fusion_interval: layer interval where cross attention layers are fused
    num_layers = max(_layer_num(k) for k in state_dict if "layers" in k) + 1
    num_fusion_layers = (
        max(_layer_num(k) for k in state_dict if "cross_attention_layers" in k) + 1
    )
    assert (
        num_layers % num_fusion_layers == 0
    ), "Conversion assumes cross attention is added at regular intervals"
    fusion_interval = num_layers // num_fusion_layers

    for key, value in state_dict.items():
        if key == "text_model.rope.freqs":
            continue
        new_key = get_mapped_key(key, _FROM_META)
        if "cross_attention_layers" in key:
            layer = int(key.split(".")[2])
            new_layer = (layer + 1) * fusion_interval - 1
            key_lst = new_key.split(".")
            key_lst[2] = str(new_layer)
            new_key = ".".join(key_lst)
        elif "conv1" in key:
            dim, flat_patch = value.shape
            patch_size = int(math.sqrt(flat_patch / 3))
            assert (
                3 * patch_size**2 == flat_patch
            ), "Conversion assumes 3 channel inputs and square patch size"
            value = value.reshape(dim, 3, patch_size, patch_size)
        converted_state_dict[new_key] = value
    return converted_state_dict


def llama3_vision_tune_to_meta(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convertor from torchtune state dict to Meta state dict. This handles:
    - Updateing the cross attention layer numbers
    - reshaping the convolution weights
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_META.items()}

    # Calculate fusion_interval: layer interval where cross attention layers are fused
    num_layers = max(_layer_num(k) for k in state_dict if "layers" in k) + 1
    num_fusion_layers = (
        max(_layer_num(k) for k in state_dict if "cross_attention_layers" in k) + 1
    )
    assert (
        num_layers % num_fusion_layers == 0
    ), "Conversion assumes cross attention is added at regular intervals"
    fusion_interval = num_layers // num_fusion_layers

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if "fusion_layer" in key:
            layer = int(key.split(".")[2])
            new_layer = (layer + 1) // fusion_interval - 1
            key_lst = new_key.split(".")
            key_lst[2] = str(new_layer)
            new_key = ".".join(key_lst)
        elif "conv" in key:
            dim = value.shape[0]
            value = value.reshape(dim, -1)
        converted_state_dict[new_key] = value
    return converted_state_dict
