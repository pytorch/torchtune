# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import nn, Tensor

from torchtune.models.llama2.attention import CausalSelfAttention
from torchtune.models.llama2.feed_forward import FeedForward
from torchtune.models.llama2.rms_norm import RMSNorm


class TransformerDecoderLayer(nn.Module):
    """
    Transformer layer used by the Llama2 model. This has a few
    differences compared to the original Transformer architecture.
        1) Uses RMSNorm instead of LayerNorm
        2) Normalization is applied before the attention and FF layer

    Args:
        self_attention (CausalSelfAttention):
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
        self_attention: CausalSelfAttention,
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


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder based on the Llama2 architecture.

    Args:
        token_embeddings (nn.Embedding):
        layer (TransformerDecoderLayer):
        num_layers (int):
        norm (RMSNorm):
        output (nn.Linear):

    TODO:
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
