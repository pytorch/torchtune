# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn, Tensor

from torchtune.modules import (
    FeedForward,
    MultiHeadAttention,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.activations import QuickGELU


class CLIPTextEncoder(nn.Module):
    """
    Text encoder for CLIP.

    Args:
        vocab_size (int): size of the vocabulary, default 49408
        max_seq_len (int): context size, default 77
        embed_dim (int): embedding/model dimension size, default 768
        num_heads (int): number of attention heads, default 12
        num_layers (int): number of transformer layers, default 12
        norm_eps (float): small value added to denominator for numerical stability, default 1e-5
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        max_seq_len: int = 77,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.empty(max_seq_len, embed_dim))

        self.encoder = nn.Sequential(
            *[
                TransformerSelfAttentionLayer(
                    attn=MultiHeadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_kv_heads=num_heads,
                        head_dim=embed_dim // num_heads,
                        q_proj=nn.Linear(embed_dim, embed_dim),
                        k_proj=nn.Linear(embed_dim, embed_dim),
                        v_proj=nn.Linear(embed_dim, embed_dim),
                        output_proj=nn.Linear(embed_dim, embed_dim),
                    ),
                    mlp=FeedForward(
                        gate_proj=nn.Linear(embed_dim, embed_dim * 4),
                        down_proj=nn.Linear(embed_dim * 4, embed_dim),
                        activation=QuickGELU(),
                    ),
                    sa_norm=nn.LayerNorm(embed_dim, eps=norm_eps),
                    mlp_norm=nn.LayerNorm(embed_dim, eps=norm_eps),
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape ``[b x s]``

        Returns:
            Tensor: output tensor with shape [b x d]

        Raises:
            ValueError: if seq_len of tokens is bigger than max_seq_len

        Shape notation:
            - b: batch size
            - s: token sequence length
            - d: token embed dim
        """
        # Input validation
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # Input embedding [b, s] -> [b, s, d]
        x = self.token_embedding(tokens) + self.position_embedding

        # Encoder [b, s, d] -> [b, s, d]
        x = self.encoder(x)
        x = self.final_norm(x)

        # Select the output of the EOS token for each encoding in the batch
        # [b, s, d] -> [b, d]
        eos_token_positions = tokens.argmax(dim=-1)
        x = x[torch.arange(bsz, device=x.device), eos_token_positions]

        return x
