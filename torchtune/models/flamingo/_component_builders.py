# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch import nn

from torchtune.models.llama3._component_builders import llama3_mlp
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.modules.model_fusion import FusionEmbedding, FusionLayer

from torchtune.modules import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TanhGate,
    TransformerCrossAttentionLayer,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

"""
Component builders for the Flamingo model and it's constituant models.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``GroupedQueryAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


# ------------------ Vanilla Flamingo ------------------

def flamingo_decoder(
    vocab_size: int,
    num_layers: int,
    fusion_interval: int,
    num_special_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3 model with additonal fused
    cross attention layers. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - Fused cross attention layers every fusion_interval number of layers
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        fusion_interval (int): interval number of layers between fusion layers
        num_special_tokens (int): number of special tokens added for the fusion model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.

    Returns:
        TransformerDecoder: Instantiation of Llama3 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = intermediate_dim or scale_hidden_dim_for_mlp(embed_dim)
    layers = []
    for idx in range(1, num_layers + 1):
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
        self_attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            attn_dropout=attn_dropout,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            attn_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        if idx % fusion_interval == 0:
            attn = GroupedQueryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                q_norm=RMSNorm(dim=head_dim, eps=1e-05),
                k_norm=RMSNorm(dim=head_dim, eps=1e-05),
                pos_embeddings=None,
                default_causal_mask=False,
                attn_dropout=0.0,
            )
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                attn_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
                attn_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            layer = FusionLayer(layer=layer, fusion_layer=xattn_layer)
        layers.append(layer)

    tok_embeddings = FusionEmbedding(vocab_size, num_special_tokens, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layers,
        num_layers=None,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
