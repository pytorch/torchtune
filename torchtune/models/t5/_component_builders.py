# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torch import nn

from torchtune.models.t5._encoder import (
    T5Encoder,
    T5EncoderLayer,
    T5EncoderSelfAttention,
)
from torchtune.modules.feed_forward import FeedForward
from torchtune.modules.rms_norm import RMSNorm


def t5_encoder(
    embed_dim: int,
    mlp_dim: int,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    rel_pos_num_buckets: int,
    rel_pos_max_dist: int,
    vocab_size: int,
    norm_eps: float,
    max_seq_len: int,
):
    """
    Builder for the T5 encoder.

    T5 paper: https://arxiv.org/abs/1910.10683

    Args:
        embed_dim (int): The model dimension.
        mlp_dim (int): The inner dimension of the feed forward layers.
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of the attention heads (should equal `embed_dim // num_heads`)
        num_layers (int): Number of encoder layers.
        rel_pos_num_buckets (int): Number of discrete buckets to divide the relative positions into.
            See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`
        rel_pos_max_dist (int): Maximum distance for relative positions.
            Distances beyond this are grouped into the last bucket.
            See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`
        vocab_size (int): Vocab size of the model's tokenizer.
        norm_eps (float): Small value added to denominator for numerical stability.
        max_seq_len (int): The maximum sequence length (context length) of the model.

    Returns:
        T5Encoder
    """
    token_embedding = nn.Embedding(vocab_size, embed_dim)

    layers = nn.ModuleList()
    for _ in range(num_layers):
        attn = T5EncoderSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            k_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            v_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        )

        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, mlp_dim, bias=False),
            down_proj=nn.Linear(mlp_dim, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, mlp_dim, bias=False),
            activation=nn.GELU(),
        )

        layer = T5EncoderLayer(
            attn=attn,
            mlp=mlp,
            sa_norm=RMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    final_norm = RMSNorm(embed_dim, eps=norm_eps)

    return T5Encoder(
        token_embedding=token_embedding,
        layers=layers,
        final_norm=final_norm,
        num_heads=num_heads,
        rel_pos_num_buckets=rel_pos_num_buckets,
        rel_pos_max_dist=rel_pos_max_dist,
        max_seq_len=max_seq_len,
    )
