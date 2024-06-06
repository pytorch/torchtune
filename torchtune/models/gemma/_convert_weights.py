# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.convert_weights import _FROM_HF, get_mapped_key


def gemma_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 8,
    num_kv_heads: int = 1,
    dim: int = 2048,
    head_dim: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a Gemma model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but doesn't load
    output projection weights.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model. Defaults to 8.
        num_kv_heads (int): Number of heads in the key/value projection layers. Defaults to 1.
        dim (int): Dimension of the model. Defaults to 2048.
        head_dim (int): Dimension of the attention head. This value is explicit in Gemma confs. Defaults to 256.

    Returns:
        Dict[str, torch.Tensor]: State dict in TorchTune's format.
    """
    converted_state_dict = {}

    def _permute(t, n_heads):
        return (
            t.view(n_heads, 2, head_dim // 2, dim)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), dim)
        )

    for key, value in state_dict.items():
        if (
            "rotary_emb.inv_freq" not in key and "lm_head.weight" not in key
        ):  # Skip loading the position embeddings and output projection weights
            new_key = get_mapped_key(key, _FROM_HF)
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)
            converted_state_dict[new_key] = value
    return converted_state_dict


def gemma_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 8,
    num_kv_heads: int = 1,
    dim: int = 2048,
    head_dim: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from TorchTune's format to Hugging Face's format for Gemma.

    This function takes a state dictionary in TorchTune's format, which contains the weights of a Gemma model,
    and converts it into a format that can be loaded into a Hugging Face model.
    The logic is identical to :func:`~torchtune.models.convert_weights.tune_to_hf`, but saves the tied
    output projection weights.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in TorchTune's format.
        num_heads (int, optional): Number of heads in the model. Defaults to 8.
        num_kv_heads (int, optional): Number of heads in the key/value projection layers. Defaults to 1.
        dim (int, optional): Dimension of the model. Defaults to 2048.
        head_dim (int): Dimension of the attention head. This value is explicit in Gemma confs. Defaults to 256.

    Returns:
        Dict[str, torch.Tensor]: State dict in Hugging Face's format.

    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

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
        elif "tok_embeddings" in key:
            # HF also uses tied weights, see
            # https://github.com/huggingface/transformers/blob/14ff5dd962c1bd0a4e3adaac347ba396d8df5add/src/transformers/models/gemma/convert_gemma_weights_to_hf.py#L104
            converted_state_dict["lm_head.weight"] = value
        converted_state_dict[new_key] = value
    return converted_state_dict
