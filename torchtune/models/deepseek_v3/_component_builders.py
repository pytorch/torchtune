# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn
from torchtune.models.deepseek_v3._linear import DeepSeekV3LatentLinear
from torchtune.models.deepseek_v3._attention import DeepSeekV3Attention
from torchtune.models.deepseek_v3._moe import DeepSeekV3TokenChoiceTopKRouter
from torchtune.modules import (
    FeedForward,
    RMSNorm,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.moe.experts import GroupedExperts
from torchtune.modules.moe.moe import MoE
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings


def deepseek_v3(
    *,
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    max_seq_len: int,
    rope_base: int = 10_000,
    q_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    moe_every_n_layers: Optional[int] = None,
    first_moe_layer: Optional[int] = None,
    num_experts: Optional[int] = None,
    num_groups: Optional[int] = None,
    topk_groups: Optional[int] = None,
    norm_topk_prob: Optional[float] = None,
    routed_scaling_factor: Optional[float] = None,
    experts_per_token: Optional[float] = None,
    mlp_hidden_dim: Optional[int] = None,
    moe_hidden_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
):
    head_dim = embed_dim // num_heads
    rope = nn.Identity()
    layers = []
    for i in range(num_layers):

        # q is sometimes decomposed into A/B (if q_lora_rank)
        # kv is *always* decomposed

        # when q is decomposed the norm is applied but
        # not otherwise - in this case the norm
        # should be applied after q a proj and before q b proj

        # for kv decomposition pos embeddings need to be extracted before
        # projecting back up
        q_head_dim = qk_rope_head_dim + qk_nope_head_dim
        if q_lora_rank is None:
            q_proj = nn.Linear(embed_dim, num_heads * q_head_dim, bias=False)
        else:
            q_proj = DeepSeekV3LatentLinear(
                in_dim=embed_dim,
                out_dim=num_heads * q_head_dim,
                rank=q_lora_rank,
                norm=RMSNorm(dim=q_lora_rank),
            )
        self_attn = DeepSeekV3Attention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            qk_rope_head_dim=head_dim,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=head_dim,
            q_head_dim=head_dim,
            q_proj=q_proj,
            kv_proj=DeepSeekV3LatentLinear(in_dim=embed_dim,
                                           out_dim=num_heads * (q_head_dim - qk_rope_head_dim + v_head_dim),
                                           rank=kv_lora_rank,
                                           norm=RMSNorm(dim=kv_lora_rank),
                                           rope_head_dim=qk_rope_head_dim),
            output_proj=nn.Linear(num_heads * v_head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            is_causal=True,
            attn_dropout=0.0,
        )
        is_moe = (moe_every_n_layers is None or (i + 1) % moe_every_n_layers == 0) and i >= first_moe_layer
        if is_moe:
            mlp_layer = MoE(
                experts=GroupedExperts(
                    dim=embed_dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                ),
                router=DeepSeekV3TokenChoiceTopKRouter(
                    gate=nn.Linear(embed_dim, num_experts, bias=False),
                    dim=embed_dim,
                    num_experts=num_experts,
                    experts_per_token=experts_per_token,
                    num_groups=num_groups,
                    topk_groups=topk_groups,
                    norm_topk_prob=norm_topk_prob,
                    routed_scaling_factor=routed_scaling_factor,
                ),
                shared_expert=deepseek_v3_mlp(embed_dim, moe_hidden_dim),
            )
        else:
            mlp_layer = deepseek_v3_mlp(embed_dim, mlp_hidden_dim)

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp_layer,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    layers = nn.ModuleList(layers)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        output=output_proj,
    )


def deepseek_v3_mlp(
    dim: int,
    hidden_dim: int
) -> FeedForward:
    """
    Builds the FeedForward layer for DeepSeek V3.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    return FeedForward(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)
