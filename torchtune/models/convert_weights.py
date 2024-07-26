# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

from typing import Any, Dict

import torch


# state dict key mappings from Meta's format to torchtune's format
_FROM_META = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        if "layers" in key:
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def meta_to_tune(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from Meta's format to torchtune's format. State dicts
    from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Eg of Meta-format state dict can be found in the ``meta-llama/Llama-2-7b``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b).

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key not in ["rope.freqs"]:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, _FROM_META)
            converted_state_dict[new_key] = value

    return converted_state_dict


def tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict


def hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to torchtune's format. State dicts
    from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Eg of HF-format state dict can be found in the ``meta-llama/Llama-2-7b-hf``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b-hf).

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
            new_key = get_mapped_key(key, _FROM_HF)
            if "q_proj" in key:
                value = _permute(value, num_heads)
            elif "k_proj" in key:
                value = _permute(value, num_kv_heads)

            converted_state_dict[new_key] = value
    return converted_state_dict


def tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
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

    Returns:
        Dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

    if head_dim is None:
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


# Mapping from torchtune LoRA module names to PEFT LoRA module names
_TO_PEFT_KEYS = {
    "lora_a": "lora_A",
    "lora_b": "lora_B",
}

# Mapping from torchtune module names to target modules for PEFT adapter config
_TO_PEFT_TARGET_MODULES = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "output_proj": "o_proj",
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
    "output": "lm_head",
}

# Keys expected in PEFT's adapter_config.json
_PEFT_CONFIG_EXPECTED_KEYS = ["target_modules", "r", "lora_alpha"]


def tune_to_peft_adapter_config(
    adapter_config: Dict[str, Any],
):
    if not all([x in adapter_config.keys() for x in _PEFT_CONFIG_EXPECTED_KEYS]):
        raise ValueError(
            f"PEFT adapter config requires {_PEFT_CONFIG_EXPECTED_KEYS}, found {adapter_config.keys()}"
        )

    for k in adapter_config["target_modules"]:
        if k not in _TO_PEFT_TARGET_MODULES:
            raise ValueError(f"Unknown target module {k}")
    adapter_config["target_modules"] = list(
        map(_TO_PEFT_TARGET_MODULES.get, adapter_config["target_modules"])
    )

    return adapter_config


def tune_to_peft_adapter_weights(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
):
    converted_state_dict = {}
    full_mapping = {}
    # Rather than recreate a separate mapping for LoRA adapter weights, we just
    # re-use the _FROM_HF mapping for base model weights. We iterate over it twice:
    # once to add mappings for LoRA A matrices and once to add mappings for LoRA B matrices.
    for k, v in _TO_PEFT_KEYS.items():
        full_mapping.update(
            {
                vv.replace(".weight", f".{k}.weight"): kk.replace(
                    ".weight", f".{v}.weight"
                )
                for kk, vv in _FROM_HF.items()
                if vv is not None
            }
        )

    if head_dim is None:
        head_dim = dim // num_heads

    def _permute_lora_matrix(t, n_heads):
        rank = t.shape[-1]
        return (
            t.view(n_heads, head_dim // 2, 2, rank)
            .transpose(1, 2)
            .reshape((head_dim * n_heads), rank)
        )

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, full_mapping)
        if "q_proj" in new_key and "lora_B" in new_key:
            value = _permute_lora_matrix(value, num_heads)
        elif "k_proj" in new_key and "lora_B" in new_key:
            value = _permute_lora_matrix(value, num_kv_heads)
        converted_state_dict["base_model.model." + new_key] = value
    return converted_state_dict
