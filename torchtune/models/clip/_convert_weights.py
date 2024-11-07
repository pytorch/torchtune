# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.convert_weights import get_mapped_key

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "text_model.embeddings.token_embedding.weight": "token_embedding.weight",
    "text_model.embeddings.position_embedding.weight": "position_embedding",
    "text_model.encoder.layers.{}.layer_norm1.weight": "encoder.{}.sa_norm.weight",
    "text_model.encoder.layers.{}.layer_norm1.bias": "encoder.{}.sa_norm.bias",
    "text_model.encoder.layers.{}.layer_norm2.weight": "encoder.{}.mlp_norm.weight",
    "text_model.encoder.layers.{}.layer_norm2.bias": "encoder.{}.mlp_norm.bias",
    "text_model.encoder.layers.{}.mlp.fc1.weight": "encoder.{}.mlp.w1.weight",
    "text_model.encoder.layers.{}.mlp.fc1.bias": "encoder.{}.mlp.w1.bias",
    "text_model.encoder.layers.{}.mlp.fc2.weight": "encoder.{}.mlp.w2.weight",
    "text_model.encoder.layers.{}.mlp.fc2.bias": "encoder.{}.mlp.w2.bias",
    "text_model.encoder.layers.{}.self_attn.q_proj.weight": "encoder.{}.attn.q_proj.weight",
    "text_model.encoder.layers.{}.self_attn.q_proj.bias": "encoder.{}.attn.q_proj.bias",
    "text_model.encoder.layers.{}.self_attn.k_proj.weight": "encoder.{}.attn.k_proj.weight",
    "text_model.encoder.layers.{}.self_attn.k_proj.bias": "encoder.{}.attn.k_proj.bias",
    "text_model.encoder.layers.{}.self_attn.v_proj.weight": "encoder.{}.attn.v_proj.weight",
    "text_model.encoder.layers.{}.self_attn.v_proj.bias": "encoder.{}.attn.v_proj.bias",
    "text_model.encoder.layers.{}.self_attn.out_proj.bias": "encoder.{}.attn.output_proj.bias",
    "text_model.encoder.layers.{}.self_attn.out_proj.weight": "encoder.{}.attn.output_proj.weight",
    "text_model.final_layer_norm.weight": "final_norm.weight",
    "text_model.final_layer_norm.bias": "final_norm.bias",
}

_IGNORE = {
    "logit_scale",
    "text_model.embeddings.position_ids",
    "text_projection.weight",
    "visual_projection.weight",
}


def clip_text_hf_to_tune(state_dict):
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("vision_model.") or key in _IGNORE:
            continue
        new_key = get_mapped_key(key, _FROM_HF)
        converted_state_dict[new_key] = value
    return converted_state_dict
