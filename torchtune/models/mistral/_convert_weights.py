# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

_MISTRAL_REWARD = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "score.weight": "output.weight",
}


def mistral_reward_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to torchtune's format, which contains the weights
    of a Mistral reward model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but with a different mapping.

    Eg of HF-format state dict can be found in the ``Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback``
    repo in HF.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" not in key:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, _MISTRAL_REWARD)
        if "q_proj" in key:
            value = _permute(value, num_heads)
        elif "k_proj" in key:
            value = _permute(value, num_kv_heads)
        converted_state_dict[new_key] = value
    return converted_state_dict


def mistral_reward_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Hugging Face's format for a Mistral reward model.

    This function takes a state dictionary in torchtune's format, which contains the weights of a Mistral reward model,
    and converts it into a format that can be loaded into a Hugging Face model.
    The logic is identical to :func:`~torchtune.models.convert_weights.tune_to_hf`, but with a different mapping.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int, optional): Number of heads in the model. Defaults to 32.
        num_kv_heads (int, optional): Number of heads in the key/value projection layers. Defaults to 32.
        dim (int, optional): Dimension of the model. Defaults to 4096.

    Returns:
        Dict[str, torch.Tensor]: State dict in Hugging Face's format.

    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _MISTRAL_REWARD.items()}
    head_dim = dim // num_heads

    def _permute(t, n_heads):
        return (
            t.view(n_heads, head_dim // 2, 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        if "q_proj" in key:
            value = _permute(value, num_heads)
        elif "k_proj" in key:
            value = _permute(value, num_kv_heads)
        converted_state_dict[new_key] = value

    return converted_state_dict
