# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import nn, Tensor

from torchtune.models.llama2.attention import llama_self_attention, LlamaSelfAttention
from torchtune.models.llama2.feed_forward import FeedForward, llama_feedforward
from torchtune.models.llama2.rms_norm import RMSNorm


def llama_transformer_decoder_layer(
    embed_dim: int,
    num_heads: int,
    max_seq_len: int = 4096,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
):
    self_attention = llama_self_attention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        max_batch_size=max_batch_size,
    )
    mlp = llama_feedforward(dim=embed_dim, hidden_dim=embed_dim)
    attn_norm = RMSNorm(dim=embed_dim)
    ff_norm = RMSNorm(dim=embed_dim)
    return TransformerDecoderLayer(
        self_attention=self_attention,
        mlp=mlp,
        attn_norm=attn_norm,
        ff_norm=ff_norm,
    )


class TransformerDecoderLayer(nn.Module):
    """
    Transformer layer used by the Llama2 model. This has a few
    differences compared to the original Transformer architecture.
        1) Uses RMSNorm instead of LayerNorm
        2) Normalization is applied before the attention and FF layer

    Args:
        self_attention (LlamaSelfAttention):
        feedforward (FeedForward):
        self_attention_norm (RMSNorm):
        feedforward_norm (RMSNorm):

    Implementation Note:
        Arg values (eg: attn_dropout) are checked for correctness (eg: belongs to [0,1])
        in the module where they are used. This helps reduces the number of ```raise```
        statements in code and improves readability.
    """

    def __init__(
        self,
        self_attention: LlamaSelfAttention,
        feedforward: FeedForward,
        self_attention_norm: RMSNorm,
        feedforward_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        # norm applied before self-attention
        self.attn_norm = self_attention_norm
        self.attn = self_attention

        # norm applied before the feedforward layer
        self.ff_norm = feedforward_norm
        self.mlp = feedforward

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        curr_pos: int = 0,
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[Tensor]): mask tensor, defaults to None.
            curr_pos (int): current position in the seq, defaults to 0.

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
        attn_out = self.attn(self.attn_norm(x), mask, curr_pos)

        # residual connection; shape: [b, s, d]
        h = attn_out + x

        mlp_out = self.mlp(self.ff_norm(h))

        # residual connection; shape: [b, s, d]
        out = h + mlp_out
        return out


def llama_transformer_decoder(
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
    max_batch_size: Optional[int] = None,
):
    token_embeddings = nn.Embedding(vocab_size, embed_dim)
    layer = TransformerDecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        max_batch_size=self.max_batch_size,
    )
    norm = RMSNorm(embed_dim, eps=norm_eps)
    output = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        token_embeddings=token_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=norm,
        output=output,
    )


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder used by the Llama2 model. This has a few
    differences compared to the original Transformer architecture.
        1) Uses RMSNorm instead of LayerNorm
        2) Normalization is applied before the attention and FF layer

    Args:
        token_embeddings (nn.Embedding):
        layer (TransformerDecoderLayer):
        num_layers (int):
        norm (RMSNorm):
        output (nn.Linear):

    TODO: A few TODOs
            - Make norm configurable
            - Make application of RoPE configurable
    """

    def __init__(
        self,
        token_embeddings: nn.Embedding,
        layer: TransformerDecoderLayer,
        num_layers: int,
        norm: RMSNorm,
        output: nn.Linear,
    ) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size

        self.max_seq_len = max_seq_len
        self.tok_embeddings = token_embeddings

        self.layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(layer)

        self.norm = norm
        self.output = output

    def forward(self, tokens: Tensor, curr_pos: int = 0) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape
                [batch_size x seq_length]
            curr_pos (int): current position in the seq, defaults to 0.
                Only relevant when incrementally decoding.

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x vocab_size]

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        # TODO: write shape
        mask = None
        if seq_len > 1 and self.max_batch_size is not None:
            mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=curr_pos + 1).type_as(h)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask, curr_pos)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = self.output(h).float()
        return output
