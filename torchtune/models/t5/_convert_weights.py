# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.models.convert_weights import get_mapped_key

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    # emb
    "encoder.embed_tokens.weight": "token_embedding.weight",
    "encoder.block.{}.layer._0.SelfAttention.relative_attention_bias.weight": "relative_position_bias.embedding.weight",
    # attn
    "encoder.block.{}.layer._0.SelfAttention.q.weight": "layers.{}.attn.q_proj.weight",
    "encoder.block.{}.layer._0.SelfAttention.k.weight": "layers.{}.attn.k_proj.weight",
    "encoder.block.{}.layer._0.SelfAttention.v.weight": "layers.{}.attn.v_proj.weight",
    "encoder.block.{}.layer._0.SelfAttention.o.weight": "layers.{}.attn.output_proj.weight",
    # ff
    "encoder.block.{}.layer._1.DenseReluDense.wi_0.weight": "layers.{}.mlp.w1.weight",
    "encoder.block.{}.layer._1.DenseReluDense.wo.weight": "layers.{}.mlp.w2.weight",
    "encoder.block.{}.layer._1.DenseReluDense.wi_1.weight": "layers.{}.mlp.w3.weight",
    # norm
    "encoder.block.{}.layer._0.layer_norm.weight": "layers.{}.sa_norm.scale",
    "encoder.block.{}.layer._1.layer_norm.weight": "layers.{}.mlp_norm.scale",
    "encoder.final_layer_norm.weight": "final_norm.scale",
}

_IGNORE = {
    "shared.weight",
    "lm_head.weight",
}


def t5_encoder_hf_to_tune(state_dict):
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("decoder.") or key in _IGNORE:
            continue

        # NOTE: HF's T5 has ".<integer>." parts that we do NOT want to be dynamically mapped
        # to corresponding ".<integer>." parts in our converted state dict.
        # This breaks the `get_mapped_key` implementation, so as a temporary hack,
        # we add leading underscores to these parts here and in the `_FROM_HF` map above.
        key = key.replace("layer.0.", "layer._0.").replace("layer.1.", "layer._1.")

        new_key = get_mapped_key(key, _FROM_HF)
        converted_state_dict[new_key] = value
    return converted_state_dict
