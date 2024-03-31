# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Literal, Optional

from torch import nn

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    KVCache,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torchtune.modules.peft import LORA_ATTN_MODULES, LoRALinear
from torchtune.models.gemma._model_utils import tie_weight

def gemma(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
) -> TransformerDecoder:
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_att = CausalSelfAttention(  # TODO: check is it the same as in the paper, SdpaAttention
        embed_dim=embed_dim,    # 2048
        num_heads=num_heads,    # 8
        num_kv_heads=num_kv_heads,  # 1
        head_dim=head_dim,  # 256
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
        pos_embeddings=rope,    # TODO: check is it the same as in the paper, GemmaRotaryEmbedding
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerDecoderLayer(
        attn=self_att,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
    # tie_weight(model)
    return model


def gemma_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    activation = nn.GELU(approximate="tanh")
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)


def lora_gemma(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # gemma args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
) -> TransformerDecoder:
    self_attn = lora_gemma_self_attention(
        lora_modules=lora_attn_modules,
        head_dim=head_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        rope_base=rope_base,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    if apply_lora_to_mlp:
        mlp = lora_gemma_mlp(
            dim=embed_dim,
            hidden_dim=intermediate_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    else:
        mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = (
        LoRALinear(
            embed_dim,
            vocab_size,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, vocab_size, bias=False)
    )
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
    # tie_weight(model)
    return model


def lora_gemma_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # CausalSelfAttention args
    head_dim: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 10_000,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
) -> CausalSelfAttention:
    if not lora_modules:
        raise ValueError(
            f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules"
        )
    q_proj = (
        LoRALinear(
            embed_dim,
            num_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "q_proj" in lora_modules
        else nn.Linear(embed_dim, num_heads * head_dim, bias=False)
    )
    k_proj = (
        LoRALinear(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "k_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    v_proj = (
        LoRALinear(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "v_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    output_proj = (
        LoRALinear(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "output_proj" in lora_modules
        else nn.Linear(embed_dim, embed_dim, bias=False)
    )
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = CausalSelfAttention(    # TODO: check is it the same as in the paper, SdpaAttention
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,    # TODO: check is it the same as in the paper, GemmaRotaryEmbedding
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_gemma_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> FeedForward:
    gate_proj = LoRALinear(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    down_proj = LoRALinear(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    up_proj = LoRALinear(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    activation = nn.GELU(approximate="tanh")
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)
