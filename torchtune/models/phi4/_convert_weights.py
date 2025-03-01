# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch

from torchtune.models.convert_weights import get_mapped_key
from torchtune.models.phi3._convert_weights import _phi_hf_to_tune, _phi_tune_to_hf


_PHI4_MINI = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.qkv_proj.base_layer.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.o_proj.base_layer.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.mlp.gate_up_proj.base_layer.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.down_proj.base_layer.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
}

def phi4_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: Optional[int],
    num_kv_heads: Optional[int],
    dim: Optional[int],
) -> Dict[str, torch.Tensor]:
    return _phi_hf_to_tune(
        state_dict,
        num_heads,
        num_kv_heads,
        dim,
        _PHI4_MINI,
    )


def phi4_tune_to_hf(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return _phi_tune_to_hf(
        state_dict,
        _PHI4_MINI,
    )

