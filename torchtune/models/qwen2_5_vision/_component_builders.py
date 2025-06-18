# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Callable
from torch import nn

from torchtune.models.qwen2_5_vision._encoder import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionTransformer,
)
from torchtune.modules import (
    MultiHeadAttention,
    RMSNorm,
    TransformerSelfAttentionLayer,
)



"""
Component builders for the Qwen 2.5 VL model and its constituent models.
torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``GroupedQueryAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def qwen2_5_vision_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    activation: Callable = nn.SiLU,
    mlp_bias: bool = True,
) -> Qwen2_5_VisionMLP:
    gate_proj = nn.Linear(in_dim, hidden_dim, bias=mlp_bias)
    down_proj = nn.Linear(hidden_dim, out_dim, bias=mlp_bias)
    up_proj = nn.Linear(hidden_dim, out_dim, bias=mlp_bias)
    return Qwen2_5_VisionMLP(
        gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation
    )


def qwen2_5_vision_encoder(
    embed_dim: int,
    num_layers: int,
    activation: Callable,
    intermediate_size: int,
    num_heads: int,
    in_channels: int,
    out_hidden_size: int,
    patch_size: int,
    spatial_merge_size: int,
    spatial_patch_size: int, # TODO: see where used
    window_size: int,
    fullatt_block_indexes: List[int],
    temporal_patch_size: int,
) -> Qwen2_5_VisionTransformer:
    """
    {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
    TODO: docstring
    Raises:
        AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    # TODO: change
    rope = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
    attn_bias = True

    # transformer layer # TODO: check if need custom attn
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        pos_embeddings=rope,
        attn_dropout=0.0,
        is_causal=False,
    )
    mlp = qwen2_5_vision_mlp( #TODO: check params
        in_dim=embed_dim,
        hidden_dim=intermediate_size,
        out_dim=embed_dim,
        activation=activation(),
        mlp_bias=True,
    )
    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=1e-6),
        mlp_norm=RMSNorm(embed_dim, eps=1e-6),
        sa_scale=None,
        mlp_scale=None,
    )
    
    patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    merger = Qwen2_5_VLPatchMerger(
        dim=out_hidden_size,
        context_dim=embed_dim,
        spatial_merge_size=spatial_merge_size,
    )

    # TODO: position embeddings
    token_pos_embedding = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    return Qwen2_5_VisionTransformer(
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        fullatt_block_indexes=fullatt_block_indexes,
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        patch_embed=patch_embed,
        patch_merger=merger,
    )