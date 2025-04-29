# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

_FROM_META = {
    "tok_embeddings.weight": "decoder.tok_embeddings.weight",
    "norm.weight": "decoder.norm.scale",
    "output.weight": "decoder.output.weight",
    "layers.{}.attention_norm.weight": "decoder.layers.{}.sa_norm.scale",
    # TODO: this is an alternative of previous line
    "layers.{}.attention.wqkv.layer_norm_weight": "decoder.layers.{}.sa_norm.scale",
    "layers.{}.attention.wq.weight": "decoder.layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wk.weight": "decoder.layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wv.weight": "decoder.layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "decoder.layers.{}.attn.output_proj.weight",
    "layers.{}.feed_forward.norm.weight": "decoder.layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.router_DE": "decoder.layers.{}.mlp.router.gate.weight",
    "layers.{}.feed_forward.w_in_shared_FD.weight": "decoder.layers.{}.mlp.shared_expert.w1.weight",
    "layers.{}.feed_forward.w_out_shared_DF.weight": "decoder.layers.{}.mlp.shared_expert.w2.weight",
    "layers.{}.feed_forward.w_swiglu_FD.weight": "decoder.layers.{}.mlp.shared_expert.w3.weight",
    "layers.{}.feed_forward.experts.moe_w_in_eD_F": "decoder.layers.{}.mlp.experts.gate_proj",
    "layers.{}.feed_forward.experts.moe_w_out_eF_D": "decoder.layers.{}.mlp.experts.down_proj",
    "layers.{}.feed_forward.experts.moe_w_swiglu_eD_F": "decoder.layers.{}.mlp.experts.up_proj",
    "layers.{}.feed_forward.global_gate_stats_3E": "decoder.layers.{}.mlp.global_gate_stats",
    "layers.{}.feed_forward.running_gate_stats_3E": "decoder.layers.{}.mlp.running_gate_stats",
    "vision_embeddings.vision_encoder.positional_embedding_vlm": "encoders.vision.clip.token_pos_embedding.positional_embedding",
    "vision_embeddings.vision_encoder.ln_pre.weight": "encoders.vision.clip.ln_pre.weight",
    "vision_embeddings.vision_encoder.ln_pre.bias": "encoders.vision.clip.ln_pre.bias",
    "vision_embeddings.vision_encoder.ln_post.weight": "encoders.vision.clip.ln_post.weight",
    "vision_embeddings.vision_encoder.ln_post.bias": "encoders.vision.clip.ln_post.bias",
    "vision_embeddings.vision_encoder.class_embedding": "encoders.vision.clip.cls_token_embedding.weight",
    "vision_embeddings.vision_encoder.conv1._linear.weight": "encoders.vision.clip.conv.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wq.weight": "encoders.vision.clip.layers.{}.attn.q_proj.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wk.weight": "encoders.vision.clip.layers.{}.attn.k_proj.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wv.weight": "encoders.vision.clip.layers.{}.attn.v_proj.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wo.weight": "encoders.vision.clip.layers.{}.attn.output_proj.weight",  # noqa
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wq.bias": "encoders.vision.clip.layers.{}.attn.q_proj.bias",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wk.bias": "encoders.vision.clip.layers.{}.attn.k_proj.bias",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wv.bias": "encoders.vision.clip.layers.{}.attn.v_proj.bias",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.attn.wo.bias": "encoders.vision.clip.layers.{}.attn.output_proj.bias",  # noqa
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.mlp.c_fc.weight": "encoders.vision.clip.layers.{}.mlp.w1.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.mlp.c_fc.bias": "encoders.vision.clip.layers.{}.mlp.w1.bias",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.mlp.c_proj.weight": "encoders.vision.clip.layers.{}.mlp.w2.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.mlp.c_proj.bias": "encoders.vision.clip.layers.{}.mlp.w2.bias",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.ln_1.weight": "encoders.vision.clip.layers.{}.sa_norm.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.ln_1.bias": "encoders.vision.clip.layers.{}.sa_norm.bias",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.ln_2.weight": "encoders.vision.clip.layers.{}.mlp_norm.weight",
    "vision_embeddings.vision_encoder.transformer.resblocks.{}.ln_2.bias": "encoders.vision.clip.layers.{}.mlp_norm.bias",
    "vision_adapter.mlp.c_fc.weight": "encoders.vision.projection.output.0.weight",
    "vision_adapter.mlp.c_proj.weight": "encoders.vision.projection.output.2.weight",
    "vision_projection.weight": "encoders.vision.projection.output.4.weight",
}

_FROM_HF = {
    "language_model.model.embed_tokens.weight": "decoder.tok_embeddings.weight",
    "language_model.model.layers.{}.self_attn.q_proj.weight": "decoder.layers.{}.attn.q_proj.weight",
    "language_model.model.layers.{}.self_attn.k_proj.weight": "decoder.layers.{}.attn.k_proj.weight",
    "language_model.model.layers.{}.self_attn.v_proj.weight": "decoder.layers.{}.attn.v_proj.weight",
    "language_model.model.layers.{}.self_attn.o_proj.weight": "decoder.layers.{}.attn.output_proj.weight",
    "language_model.model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "language_model.model.layers.{}.feed_forward.shared_expert.down_proj.weight": "decoder.layers.{}.mlp.shared_expert.w2.weight",
    "language_model.model.layers.{}.feed_forward.shared_expert.gate_proj.weight": "decoder.layers.{}.mlp.shared_expert.w1.weight",
    "language_model.model.layers.{}.feed_forward.shared_expert.up_proj.weight": "decoder.layers.{}.mlp.shared_expert.w3.weight",
    "language_model.model.layers.{}.feed_forward.experts.down_proj": "decoder.layers.{}.mlp.experts.down_proj",
    # NOTE: this one needs special handling, doing it hackily for now
    "language_model.model.layers.{}.feed_forward.experts.gate_up_proj": "decoder.layers.{}.mlp.experts.gate_proj",
    "language_model.model.layers.{}.feed_forward.router.weight": "decoder.layers.{}.mlp.router.gate.weight",
    # The next three are still needed for 128E model
    "language_model.model.layers.{}.feed_forward.gate_proj.weight": "decoder.layers.{}.mlp.w1.weight",
    "language_model.model.layers.{}.feed_forward.up_proj.weight": "decoder.layers.{}.mlp.w3.weight",
    "language_model.model.layers.{}.feed_forward.down_proj.weight": "decoder.layers.{}.mlp.w2.weight",
    "language_model.model.layers.{}.input_layernorm.weight": "decoder.layers.{}.sa_norm.scale",
    "language_model.model.layers.{}.post_attention_layernorm.weight": "decoder.layers.{}.mlp_norm.scale",
    "language_model.model.norm.weight": "decoder.norm.scale",
    "language_model.lm_head.weight": "decoder.output.weight",
    "vision_model.layernorm_pre.weight": "encoders.vision.clip.ln_pre.weight",
    "vision_model.layernorm_pre.bias": "encoders.vision.clip.ln_pre.bias",
    "vision_model.layernorm_post.weight": "encoders.vision.clip.ln_post.weight",
    "vision_model.layernorm_post.bias": "encoders.vision.clip.ln_post.bias",
    "vision_model.class_embedding": "encoders.vision.clip.cls_token_embedding.weight",
    "vision_model.positional_embedding_vlm": "encoders.vision.clip.token_pos_embedding.positional_embedding",
    "vision_model.patch_embedding.linear.weight": "encoders.vision.clip.conv.weight",
    "vision_model.model.layers.{}.self_attn.q_proj.weight": "encoders.vision.clip.layers.{}.attn.q_proj.weight",
    "vision_model.model.layers.{}.self_attn.k_proj.weight": "encoders.vision.clip.layers.{}.attn.k_proj.weight",
    "vision_model.model.layers.{}.self_attn.v_proj.weight": "encoders.vision.clip.layers.{}.attn.v_proj.weight",
    "vision_model.model.layers.{}.self_attn.o_proj.weight": "encoders.vision.clip.layers.{}.attn.output_proj.weight",
    "vision_model.model.layers.{}.self_attn.q_proj.bias": "encoders.vision.clip.layers.{}.attn.q_proj.bias",
    "vision_model.model.layers.{}.self_attn.k_proj.bias": "encoders.vision.clip.layers.{}.attn.k_proj.bias",
    "vision_model.model.layers.{}.self_attn.v_proj.bias": "encoders.vision.clip.layers.{}.attn.v_proj.bias",
    "vision_model.model.layers.{}.self_attn.o_proj.bias": "encoders.vision.clip.layers.{}.attn.output_proj.bias",
    "vision_model.model.layers.{}.mlp.fc1.weight": "encoders.vision.clip.layers.{}.mlp.w1.weight",
    "vision_model.model.layers.{}.mlp.fc1.bias": "encoders.vision.clip.layers.{}.mlp.w1.bias",
    "vision_model.model.layers.{}.mlp.fc2.weight": "encoders.vision.clip.layers.{}.mlp.w2.weight",
    "vision_model.model.layers.{}.mlp.fc2.bias": "encoders.vision.clip.layers.{}.mlp.w2.bias",
    "vision_model.model.layers.{}.input_layernorm.weight": "encoders.vision.clip.layers.{}.sa_norm.weight",
    "vision_model.model.layers.{}.input_layernorm.bias": "encoders.vision.clip.layers.{}.sa_norm.bias",
    "vision_model.model.layers.{}.post_attention_layernorm.weight": "encoders.vision.clip.layers.{}.mlp_norm.weight",
    "vision_model.model.layers.{}.post_attention_layernorm.bias": "encoders.vision.clip.layers.{}.mlp_norm.bias",
    "vision_model.vision_adapter.mlp.fc1.weight": "encoders.vision.projection.output.0.weight",
    "vision_model.vision_adapter.mlp.fc2.weight": "encoders.vision.projection.output.2.weight",
    "multi_modal_projector.linear_1.weight": "encoders.vision.projection.output.4.weight",
}

_IGNORE = {
    # No need to overwrite our precomputed rope cache for training
    "rope.freqs",
    # Not needed for vision_adapter(pixel shuffle) fc as it's identity
    "vision_adapter.fc.weight",
}


def llama4_meta_to_tune(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convertor from Meta state dict to torchtune state dict. This handles:
    - skip loading weights from the _IGNORE list
    - skip loading the expert activation stats used in MOE inference
    - transpose the weight for router
    - transpose and unsqueeze the weight for shared experts
    - unflatten the expert dimension for experts weights
    - reshaping the convolution weights
    """
    converted_state_dict = {}
    num_experts = state_dict["layers.0.feed_forward.router_DE"].shape[1]
    for key, value in state_dict.items():
        # TODO: add these full key names to _IGNORE
        if key in _IGNORE or "expert_activation_DE" in key:
            continue
        elif "router_DE" in key:
            value = value.transpose(0, 1)
        elif "experts" in key:
            value = value.view(num_experts, -1, value.shape[-1])
        elif "conv1" in key:
            dim, flat_patch = value.shape
            patch_size = math.isqrt(flat_patch // 3)
            assert (
                3 * patch_size**2 == flat_patch
            ), "Conversion assumes 3 channel inputs and square patch size"
            value = value.reshape(dim, 3, patch_size, patch_size)

        new_key = get_mapped_key(key, _FROM_META)
        converted_state_dict[new_key] = value
    return converted_state_dict


def llama4_tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convertor from torchtune state dict to Meta state dict. This handles:
    - transpose the weight for router
    - squeeze the expert dimension and transpose the weight for shared experts
    - flatten the expert dimension for experts weights
    - reshaping the convolution weights
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_META.items()}

    for key, value in state_dict.items():
        # get the invert key name for vision adaptor weights, get_mapped_key will not work as it will look up for {}
        if (
            key == "encoders.vision.projection.adapter.0.weight"
            or key == "encoders.vision.projection.adapter.2.weight"
        ):
            new_key = inverted_mapping_dict[key]
        else:
            new_key = get_mapped_key(key, inverted_mapping_dict)

        if "router" in key:
            value = value.transpose(0, 1)
        elif "experts" in key:
            value = value.view(-1, value.shape[-1])
        elif "conv" in key:
            dim = value.shape[0]
            value = value.reshape(dim, -1)
        elif new_key.startswith("module.module_list") and any(
            k.isdigit() for k in new_key.split(".")
        ):
            abstract_key = re.sub(r"(\.\d+)", ".{}", new_key)
            layer_num = re.search(r"\d+", new_key).group(0)
            new_layer_num = str(int(layer_num) + 5)
            new_key = abstract_key.format(new_layer_num)
        converted_state_dict[new_key] = value
    return converted_state_dict


def llama4_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, _FROM_HF)
        if "language_model" in key:
            if "gate_up_proj" in key:
                gate_proj, up_proj = torch.chunk(value, 2, dim=-1)
                converted_state_dict[new_key] = gate_proj
                converted_state_dict[new_key.replace("gate", "up")] = up_proj
                continue
        elif "patch_embedding" in key:
            d, flat_patch = value.shape
            patch_size = math.isqrt(flat_patch // 3)
            assert (
                3 * patch_size**2 == flat_patch
            ), "Conversion assumes 3 channel inputs and square patch size"
            value = value.reshape(d, 3, patch_size, patch_size)

        converted_state_dict[new_key] = value

    return converted_state_dict


def llama4_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

    for key, value in state_dict.items():
        # Handle special cases first
        if key in {
            "encoders.vision.projection.output.0.weight",
            "encoders.vision.projection.output.2.weight",
            "encoders.vision.projection.output.4.weight",
        }:
            new_key = inverted_mapping_dict[key]
        elif key.endswith("experts.gate_proj"):
            # Combine gate projection with up projection
            new_key = get_mapped_key(key, inverted_mapping_dict)
            up_proj = state_dict[key.replace("gate", "up")]
            converted_state_dict[new_key] = torch.cat([value, up_proj], dim=-1)
            continue
        elif key.endswith("experts.up_proj"):
            # Skip as already handled with gate projection
            continue
        elif "conv" in key:
            # Reshape convolution weights
            value = value.reshape(value.shape[0], -1)
            new_key = get_mapped_key(key, inverted_mapping_dict)
        else:
            # Default case - just map the key
            new_key = get_mapped_key(key, inverted_mapping_dict)

        converted_state_dict[new_key] = value

    return converted_state_dict
