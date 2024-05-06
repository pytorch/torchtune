# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key


_PHI3_MINI = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.qkv_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.mlp.gate_up_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}


def phi3_hf_to_tune(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convertor from HF state dict to torchtune state dict. This handles:
    - Splitting the fused q,k and v matrix
    - Splitting the fused gate and up projection matrix
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, _PHI3_MINI)
        if "qkv" in key:
            (
                q,
                k,
                v,
            ) = value.chunk(3, dim=0)
            converted_state_dict[new_key] = q
            converted_state_dict[new_key.replace("q_proj", "k_proj")] = k
            converted_state_dict[new_key.replace("q_proj", "v_proj")] = v
        elif "gate" in key:
            w1, w3 = value.chunk(2, dim=0)
            converted_state_dict[new_key] = w1
            converted_state_dict[new_key.replace("w1", "w3")] = w3
        else:
            converted_state_dict[new_key] = value
    return converted_state_dict


def phi3_tune_to_hf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convertor from torchtune state dict to HF state dict. This handles:
    - Fusing q,k and v matrix
    - Fusing gate and up projection matrix
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _PHI3_MINI.items()}

    for key, value in state_dict.items():
        if "k_proj" in key or "v_proj" in key or "w3" in key:
            # these keys are accounted for separately and should be skipped
            continue
        new_key = get_mapped_key(key, inverted_mapping_dict)

        if "q_proj" in key:
            q = value
            k = state_dict[key.replace("q_proj", "k_proj")]
            v = state_dict[key.replace("q_proj", "v_proj")]
            qkv = torch.cat([q, k, v], dim=0)
            # q_proj maps to qkv_proj; no need to string replace
            converted_state_dict[new_key] = qkv

        elif "w1" in key:
            gate_proj = value
            up_proj = state_dict[key.replace("w1", "w3")]
            gate_up_proj = torch.cat([gate_proj, up_proj], dim=0)
            # w1 maps to gate_up_proj; no need to string replace
            converted_state_dict[new_key] = gate_up_proj

        else:
            converted_state_dict[new_key] = value
    return converted_state_dict
