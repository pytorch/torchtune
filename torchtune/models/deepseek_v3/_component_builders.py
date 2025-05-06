# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn
from torchtune.models.deepseek_v3._linear import DeepSeekV3LatentLinear
from torchtune.models.deepseek_v3._position_embeddings import DeepSeekV3RoPE
from torchtune.modules import (
    FeedForward,
    MultiheadAttention,
    RMSNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
    Tokenizer
)
from torchtune.modules.moe.experts import GroupedExperts
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings

def deepseek_v3_mlp(
    dim: int,
    hidden_dim: int,
    ffn_dropout: float = 0.0
) -> FeedForward:
    """
    Builds the FeedForward layer for DeepSeek V3.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    return FeedForward(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj, dropout=ffn_dropout, activation_fn=nn.SiLU)
    
def deepseek_v3(
    *,
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    rope_base: int = 10_000,
    q_lora_rank: Optional[int] = None,
    rope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
):
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    layers = []
    for i in range(num_layers):
        if q_lora_rank is None:
            q_proj = nn.Linear(embed_dim, num_heads * q_head_dim, bias=False)
        else:
            q_proj = DeepSeekV3LatentLinear(
                in_dim=embed_dim,
                out_dim=num_heads * q_head_dim,
                rank=q_lora_rank,
            )
        kv_proj = DeepSeekV3LatentLinear(
            in_dim=embed_dim,
            out_dim=num_kv_heads * (q_head_dim - rope_head_dim + v_head_dim),
            rank=q_lora_rank,
        )
        self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            pos_embeddings=rope,
            q_proj=q_proj,
            kv_proj=kv_proj,    
            o_proj=nn.Linear(num_heads * v_head_dim, embed_dim, bias=False),
        )
        if i >= first_moe_layer and i % moe_every_n_layers == 0:
            mlp_layer = MoE(
                experts=GroupedExperts(
                    dim=embed_dim, hidden_dim=hidden_dim, num_experts=num_experts
                ),
                router=TokenChoiceTopKRouter(
                    gate=nn.Linear(embed_dim, num_experts, bias=False),
                    dim=embed_dim,
                    num_experts=num_experts,
                    experts_per_token=experts_per_token,
                ),
                shared_expert=(
                    deepseek_v3_mlp(dim=embed_dim, hidden_dim=hidden_dim) if use_shared_expert else None
                )
            )
        else:
            mlp_layer = deepseek_v3_mlp(dim=embed_dim, hidden_dim=hidden_dim)

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
    )