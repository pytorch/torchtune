# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

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


_FROM_HF = {
    "language_model.model.embed_tokens.weight": "decoder.tok_embeddings.weight",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "decoder.layers.{}.attn.q_proj.weight",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "decoder.layers.{}.attn.k_proj.weight",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "decoder.layers.{}.attn.v_proj.weight",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "decoder.layers.{}.attn.output_proj.weight",
    "language_model.model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "language_model.model.layers.{}.mlp.gate_proj.weight": "decoder.layers.{}.mlp.w1.weight",
    "language_model.model.layers.{}.mlp.up_proj.weight": "decoder.layers.{}.mlp.w3.weight",
    "language_model.model.layers.{}.mlp.down_proj.weight": "decoder.layers.{}.mlp.w2.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "decoder.layers.{}.sa_norm.scale",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "decoder.layers.{}.mlp_norm.scale",
    "language_model.model.norm.weight": "decoder.norm.scale",
    "language_model.lm_head.weight": "decoder.output.weight",
    "language_model.model.layers.{}.cross_attn_attn_gate": "decoder.layers.{}.fusion_layer.ca_scale.scale",
    "language_model.model.layers.{}.cross_attn_mlp_gate": "decoder.layers.{}.fusion_layer.mlp_scale.scale",
    "language_model.model.layers.{}.cross_attn.q_proj.weight": "decoder.layers.{}.fusion_layer.attn.q_proj.weight",
    "language_model.model.layers.{}.cross_attn.k_proj.weight": "decoder.layers.{}.fusion_layer.attn.k_proj.weight",
    "language_model.model.layers.{}.cross_attn.v_proj.weight": "decoder.layers.{}.fusion_layer.attn.v_proj.weight",
    "language_model.model.layers.{}.cross_attn.o_proj.weight": "decoder.layers.{}.fusion_layer.attn.output_proj.weight",
    "language_model.model.layers.{}.cross_attn.q_norm.weight": "decoder.layers.{}.fusion_layer.attn.q_norm.scale",
    "language_model.model.layers.{}.cross_attn.k_norm.weight": "decoder.layers.{}.fusion_layer.attn.k_norm.scale",
    "vision_model.gated_positional_embedding.embedding": "encoder.clip.token_pos_embedding.local_token_positional_embedding",
    "vision_model.gated_positional_embedding.tile_embedding.weight": "encoder.clip.token_pos_embedding.global_token_positional_embedding",  # noqa
    "vision_model.gated_positional_embedding.gate": "encoder.clip.token_pos_embedding.gate",
    "vision_model.layernorm_pre.weight": "encoder.clip.ln_pre.weight",
    "vision_model.layernorm_pre.bias": "encoder.clip.ln_pre.bias",
    "vision_model.layernorm_post.weight": "encoder.clip.ln_post.weight",
    "vision_model.layernorm_post.bias": "encoder.clip.ln_post.bias",
    "vision_model.pre_tile_positional_embedding.embedding.weight": "encoder.clip.pre_tile_pos_embed.embedding",
    "vision_model.pre_tile_positional_embedding.gate": "encoder.clip.pre_tile_pos_embed.gate",
    "vision_model.post_tile_positional_embedding.embedding.weight": "encoder.clip.post_tile_pos_embed.embedding",
    "vision_model.post_tile_positional_embedding.gate": "encoder.clip.post_tile_pos_embed.gate",
    "vision_model.class_embedding": "encoder.clip.cls_token_embedding.weight",
    "vision_model.patch_embedding.weight": "encoder.clip.conv.weight",
    "vision_model.transformer.layers.{}.self_attn.q_proj.weight": "encoder.clip.layers.{}.attn.q_proj.weight",
    "vision_model.transformer.layers.{}.self_attn.k_proj.weight": "encoder.clip.layers.{}.attn.k_proj.weight",
    "vision_model.transformer.layers.{}.self_attn.v_proj.weight": "encoder.clip.layers.{}.attn.v_proj.weight",
    "vision_model.transformer.layers.{}.self_attn.o_proj.weight": "encoder.clip.layers.{}.attn.output_proj.weight",
    "vision_model.transformer.layers.{}.mlp.fc1.weight": "encoder.clip.layers.{}.mlp.w1.weight",
    "vision_model.transformer.layers.{}.mlp.fc1.bias": "encoder.clip.layers.{}.mlp.w1.bias",
    "vision_model.transformer.layers.{}.mlp.fc2.weight": "encoder.clip.layers.{}.mlp.w2.weight",
    "vision_model.transformer.layers.{}.mlp.fc2.bias": "encoder.clip.layers.{}.mlp.w2.bias",
    "vision_model.transformer.layers.{}.input_layernorm.weight": "encoder.clip.layers.{}.sa_norm.weight",
    "vision_model.transformer.layers.{}.input_layernorm.bias": "encoder.clip.layers.{}.sa_norm.bias",
    "vision_model.transformer.layers.{}.post_attention_layernorm.weight": "encoder.clip.layers.{}.mlp_norm.weight",
    "vision_model.transformer.layers.{}.post_attention_layernorm.bias": "encoder.clip.layers.{}.mlp_norm.bias",
    "vision_model.global_transformer.layers.{}.self_attn.q_proj.weight": "encoder.projection.layers.{}.attn.q_proj.weight",
    "vision_model.global_transformer.layers.{}.self_attn.k_proj.weight": "encoder.projection.layers.{}.attn.k_proj.weight",
    "vision_model.global_transformer.layers.{}.self_attn.v_proj.weight": "encoder.projection.layers.{}.attn.v_proj.weight",
    "vision_model.global_transformer.layers.{}.self_attn.o_proj.weight": "encoder.projection.layers.{}.attn.output_proj.weight",
    "vision_model.global_transformer.layers.{}.mlp.fc1.weight": "encoder.projection.layers.{}.mlp.w1.weight",
    "vision_model.global_transformer.layers.{}.mlp.fc1.bias": "encoder.projection.layers.{}.mlp.w1.bias",
    "vision_model.global_transformer.layers.{}.mlp.fc2.weight": "encoder.projection.layers.{}.mlp.w2.weight",
    "vision_model.global_transformer.layers.{}.mlp.fc2.bias": "encoder.projection.layers.{}.mlp.w2.bias",
    "vision_model.global_transformer.layers.{}.input_layernorm.weight": "encoder.projection.layers.{}.sa_norm.weight",
    "vision_model.global_transformer.layers.{}.input_layernorm.bias": "encoder.projection.layers.{}.sa_norm.bias",
    "vision_model.global_transformer.layers.{}.post_attention_layernorm.weight": "encoder.projection.layers.{}.mlp_norm.weight",
    "vision_model.global_transformer.layers.{}.post_attention_layernorm.bias": "encoder.projection.layers.{}.mlp_norm.bias",
    "vision_model.global_transformer.layers.{}.gate_attn": "encoder.projection.layers.{}.sa_scale.scale",
    "vision_model.global_transformer.layers.{}.gate_ffn": "encoder.projection.layers.{}.mlp_scale.scale",
    "multi_modal_projector.weight": "encoder.projection.output.weight",
    "multi_modal_projector.bias": "encoder.projection.output.bias",
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
    # Get the number of unique fusion layers.
    # Keys have the form decoder.fusion_layer.i. ... where i is the layer number
    num_fusion_layers = len(
        set([k.split(".")[2] for k in state_dict if "fusion_layer" in k])
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


def llama3_vision_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    vocab_size: int = 128256,
    cross_attention_layers: Optional[List[int]] = None,
    # Vision Encoder Paramters
    encoder_dim: int = 1280,
    tile_size: int = 448,
    num_tiles: int = 4,
    supported_aspect_ratios: List[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convertor from HF state dict to torchtune state dict. This handles:
    - Updating the cross attention layer numbers
    - skip loading the rope embeddings
    - reshaping q, k projections
    - reversing the precomputed vision positional embeddings
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads
    if cross_attention_layers is None:
        cross_attention_layers = []

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue
        new_key = get_mapped_key(key, _FROM_HF)
        if "language_model" in key:
            if "layers" in key:  # Update layer numbers
                layer = int(key.split(".")[3])
                num_shifts = sum(layer > l for l in cross_attention_layers)
                new_layer = layer - num_shifts
                key_lst = new_key.split(".")
                if layer in cross_attention_layers and "fusion_layer" not in new_key:
                    # some keys are the same for sa and ca, so we need to edit them here
                    key_lst[2] = f"{new_layer}.fusion_layer"
                    if "sa_norm" in new_key:
                        key_lst[3] = "ca_norm"
                else:
                    key_lst[2] = str(new_layer)
                new_key = ".".join(key_lst)
            if "q_proj" in key and "cross_attn" not in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key and "cross_attn" not in key:
                value = _permute(value, num_kv_heads)
            elif new_key == "decoder.tok_embeddings.weight":
                # Split embedding between learnable embeddings and original text embedding
                learned_embedding = "decoder.tok_embeddings.fusion_embedding.weight"
                converted_state_dict[learned_embedding] = value[vocab_size:]
                value = value[:vocab_size]
        elif "vision_model" in key:
            if (
                "tile_pos_embed.embedding" in new_key
                or "global_token_positional_embedding" in new_key
            ):
                # WARNING: META format postional embeddings contain embeddings that
                # the model can never use (4 tiles -> 4 x 4 embeddings -> a 4 x 4 image would be 16 tiles).
                # HF removes these extra embeddings, for us to convert to the META format we set those
                # unused embeddings as 0 instead of the original random (untrained) values in the original
                # META checkpoing
                num_embeds = value.shape[-1] // encoder_dim // num_tiles
                pos_embedding = torch.zeros(
                    num_tiles,
                    num_tiles,
                    num_embeds,
                    encoder_dim,
                    device=value.device,
                    dtype=value.dtype,
                )
                # Loop through aspect ratios and assign precomputed embeds back to Meta Llama embeddings
                for i, (h, w) in enumerate(supported_aspect_ratios or []):
                    if h * w == num_tiles:  # h*w < num_tiles is redundant
                        # i == 0 is used for padding in HF
                        pos_embedding[:h, :w] = value[i + 1].reshape(
                            h, w, num_embeds, encoder_dim
                        )
                value = pos_embedding

        converted_state_dict[new_key] = value
    return converted_state_dict


def llama3_vision_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    vocab_size: int = 128256,
    cross_attention_layers: Optional[List[int]] = None,
    # Vision Encoder Paramters
    encoder_dim: int = 1280,
    tile_size: int = 448,
    num_tiles: int = 4,
    supported_aspect_ratios: List[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convertor from Tune state dict to HF state dict. This handles:
    - Updateing the cross attention layer numbers
    - skip loading the rope embeddings
    - reshaping q, k projections
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
    # missing keys in _FROM_HF due to naming collisions
    missing_keys = {
        "decoder.layers.{}.fusion_layer.ca_norm.scale": "language_model.model.layers.{}.input_layernorm.weight",
        "decoder.layers.{}.fusion_layer.mlp_norm.scale": "language_model.model.layers.{}.post_attention_layernorm.weight",
        "decoder.layers.{}.fusion_layer.mlp.w1.weight": "language_model.model.layers.{}.mlp.gate_proj.weight",
        "decoder.layers.{}.fusion_layer.mlp.w3.weight": "language_model.model.layers.{}.mlp.up_proj.weight",
        "decoder.layers.{}.fusion_layer.mlp.w2.weight": "language_model.model.layers.{}.mlp.down_proj.weight",
        "decoder.tok_embeddings.fusion_embedding.weight": None,
    }
    inverted_mapping_dict.update(missing_keys)

    if head_dim is None:
        head_dim = dim // num_heads
    if cross_attention_layers is None:
        cross_attention_layers = []
    # convert hf layer numbers to tune numbers
    cross_attention_layers = [
        l - i for i, l in enumerate(sorted(cross_attention_layers))
    ]

    def _permute(t, n_heads):
        return (
            t.view(n_heads, head_dim // 2, 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if "decoder" in key:
            if "layers" in key:  # Update layer numbers
                layer = int(key.split(".")[2])
                num_shifts = sum(layer > l for l in cross_attention_layers)
                new_layer = layer + num_shifts
                key_lst = new_key.split(".")
                if layer in cross_attention_layers and "fusion_layer" not in key:
                    new_layer += 1  # hf treats the fusion_layer as an additional layer
                key_lst[3] = str(new_layer)
                new_key = ".".join(key_lst)
            if "q_proj" in key and "cross_attn" not in new_key:
                value = _permute(value, num_heads)
            elif "k_proj" in key and "cross_attn" not in new_key:
                value = _permute(value, num_kv_heads)
            elif key == "decoder.tok_embeddings.weight":
                learned_embedding = state_dict[
                    "decoder.tok_embeddings.fusion_embedding.weight"
                ]
                value = torch.cat([value, learned_embedding])
            elif key == "decoder.tok_embeddings.fusion_embedding.weight":
                continue
        elif "encoder" in key:
            if (
                "tile_pos_embed.embedding" in key
                or "global_token_positional_embedding" in key
            ):
                num_embeds = value.shape[-2]
                pos_embedding = torch.zeros(
                    len(supported_aspect_ratios) + 1,
                    num_tiles,
                    num_embeds,
                    encoder_dim,
                    device=value.device,
                    dtype=value.dtype,
                )
                # Loop through aspect ratios and precompute embeds per aspect ratio
                for i, (h, w) in enumerate(supported_aspect_ratios or []):
                    pos_embedding[i + 1, : h * w] = value[:h, :w].reshape(
                        h * w, num_embeds, encoder_dim
                    )
                value = pos_embedding.flatten(1)

        converted_state_dict[new_key] = value
    return converted_state_dict
