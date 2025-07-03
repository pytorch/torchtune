# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key
from torchtune.models.qwen2._convert_weights import _FROM_HF as _FROM_HF_QWEN2

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "visual.blocks.{}.attn.proj.bias": "visual.layers.{}.attn.output_proj.bias",
    "visual.blocks.{}.attn.proj.weight": "visual.layers.{}.attn.output_proj.weight",
    "visual.blocks.{}.attn.qkv.bias": "visual.layers.{}.attn.q_proj.bias",
    "visual.blocks.{}.attn.qkv.weight": "visual.layers.{}.attn.q_proj.weight",
    "visual.blocks.{}.mlp.down_proj.bias": "visual.layers.{}.mlp.w2.bias",
    "visual.blocks.{}.mlp.down_proj.weight": "visual.layers.{}.mlp.w2.weight",
    "visual.blocks.{}.mlp.gate_proj.bias": "visual.layers.{}.mlp.w1.bias",
    "visual.blocks.{}.mlp.gate_proj.weight": "visual.layers.{}.mlp.w1.weight",
    "visual.blocks.{}.mlp.up_proj.bias": "visual.layers.{}.mlp.w3.bias",
    "visual.blocks.{}.mlp.up_proj.weight": "visual.layers.{}.mlp.w3.weight",
    "visual.blocks.{}.norm1.weight": "visual.layers.{}.sa_norm.scale",
    "visual.blocks.{}.norm2.weight": "visual.layers.{}.mlp_norm.scale",
    "visual.merger.ln_q.weight": "visual.merger.ln_q.scale",
    "visual.merger.mlp.{}.bias": "visual.merger.mlp.{}.bias",
    "visual.merger.mlp.{}.weight": "visual.merger.mlp.{}.weight",
    "visual.patch_embed.proj.weight": "visual.patch_embed.proj.weight"
}

_FROM_HF.update(_FROM_HF_QWEN2)

QWEN2_TIED_KEY = "lm_head.weight"


def qwen2_5_vl_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    tie_word_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a Qwen2 model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but may not load
    output projection weights.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}

    for key, value in state_dict.items():
        if "qkv" in key:
            (
                q,
                k,
                v,
            ) = value.chunk(3, dim=0)
            converted_state_dict[new_key] = q
            converted_state_dict[new_key.replace("q_proj", "k_proj")] = k
            converted_state_dict[new_key.replace("q_proj", "v_proj")] = v
        elif (
            tie_word_embeddings and QWEN2_TIED_KEY in key
        ):  # Skip loading the output projection weights
            continue
        elif "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue
        else:
            new_key = get_mapped_key(key, _FROM_HF)
            converted_state_dict[new_key] = value
    return converted_state_dict


def qwen2_5_vl_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
):
    """
    Convert a state dict from torchtune's format to HF's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        Dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if "q_proj" in key:
            q = value
            k = state_dict[key.replace("q_proj", "k_proj")]
            v = state_dict[key.replace("q_proj", "v_proj")]
            qkv = torch.cat([q, k, v], dim=0)
            # q_proj maps to qkv_proj; no need to string replace
            converted_state_dict[new_key] = qkv
        else:
            converted_state_dict[new_key] = value

    return converted_state_dict
