# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from llm.llama2.attention import LlamaSelfAttention
from llm.llama2.feed_forward import FeedForward
from llm.llama2.rms_norm import RMSNorm

from torch import nn, Tensor


class TransformerDecoderLayer(nn.Module):
    """
    Transformer layer used by the Llama2 model. This has a few
    differences compared to the original Transformer architecture.
        1) Uses RMSNorm instead of LayerNorm
        2) Normalization is applied before the attention and FF layer

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. To enable MHA, set
            ```num_kv_heads``` = ```num_heads``` or ```num_kv_heads``` = None
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default value is 4096
        num_kv_heads (Optional[int]): number of key and value heads. User should
            ensure `num_heads` % `num_kv_heads` == 0. Default value is None, in
            which case this is the same as MHA
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0
        norm_eps (float): eps value of for RMSNorm. Default is 1e-5.

    Implementation Note:
        Arg values (eg: attn_dropout) are checked for correctness (eg: belongs to [0,1])
        in the module where they are used. This helps reduces the number of ```raise```
        statements in code and improves readability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 4096,
        num_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        # norm applied before self-attention
        self.attn_norm = RMSNorm(dim=embed_dim, eps=norm_eps)
        self.attn = LlamaSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )

        # norm applied before the feedforward layer
        self.ff_norm = RMSNorm(dim=embed_dim, eps=norm_eps)
        self.mlp = FeedForward(dim=embed_dim, hidden_dim=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - d: embed dim

        TODO: A few TODOs
            - Make position of norm configurable
        """
        # input tensor and attention output have the same shape
        # [b, s, d]
        attn_out = self.attn(self.attn_norm(x))

        # residual connection; shape: [b, s, d]
        h = attn_out + x

        mlp_out = self.mlp(self.ff_norm(h))

        # residual connection; shape: [b, s, d]
        out = h + mlp_out
        return out


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder used by the Llama2 model. This has a few
    differences compared to the original Transformer architecture.
        1) Uses RMSNorm instead of LayerNorm
        2) Normalization is applied before the attention and FF layer

    Args:
        vocab_size (int): Size of the vocabulary supported by the model. This
            controls the number of rows in the embedding table
        embed_dim (int): embedding dimension for the model
        num_layers (int): number of TransformerDecoderLayers
        num_heads (int): number of query heads. To enable MHA, set
            ```num_kv_heads``` = ```num_heads``` or ```num_kv_heads``` = None
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default value is 4096
        num_kv_heads (Optional[int]): number of key and value heads. User should
            ensure `num_kv_heads` % `num_heads` == 0. Default value is None, in
            which case this is the same as MHA
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0
        norm_eps (float): eps value of for RMS Norm. Default is 1e-5.

    TODO: A few TODOs
            - Make norm configurable
            - Make application of RoPE configurable
    """

    def __init__(
        self,
        # embedding params
        vocab_size: int,
        embed_dim: int,
        # transformer layer params
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 4096,
        num_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        # RMS Norm params
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_embeddings = nn.Embedding(vocab_size, embed_dim)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    max_seq_len=max_seq_len,
                    attn_dropout=attn_dropout,
                    norm_eps=norm_eps,
                )
            )

        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape
                [batch_size x seq_length]

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x vocab_size]

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = self.output(h).float()
        return output
