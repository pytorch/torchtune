# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch import nn, Tensor

from torchtune.modules import CausalSelfAttention, FeedForward, RMSNorm


class TransformerDecoderLayer(nn.Module):
    """Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        self_attention (CausalSelfAttention): Attention module.
        mlp (FeedForward): Feed-forward module.
        sa_norm (nn.Module): Normalization to be applied before self-attention.
        mlp_norm (nn.Module): Normalization to be applied before the feed-forward layer.
    """

    def __init__(
        self,
        self_attention: CausalSelfAttention,
        mlp: FeedForward,
        sa_norm: nn.Module,
        mlp_norm: nn.Module,
    ) -> None:
        super().__init__()
        # Norm applied before self-attention
        self.sa_norm = sa_norm
        self.attn = self_attention

        # Norm applied before the feedforward layer
        self.mlp_norm = mlp_norm
        self.mlp = mlp

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
        attn_out = self.attn(self.sa_norm(x), mask, curr_pos)

        # residual connection; shape: [b, s, d]
        h = attn_out + x

        mlp_out = self.mlp(self.mlp_norm(h))

        # residual connection; shape: [b, s, d]
        out = h + mlp_out
        return out


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture:
    https://arxiv.org/abs/2307.09288

    Args:
        token_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move tokens to an embedding space.
        layer (TransformerDecoderLayer): Instantiation of a single TransformerDecoderLayer, to be used in the decoder.
        num_layers (int): Number of ``layers`` in the decoder.
        norm (nn.Module): Callable that applies normalization to the output of the decoder, before final MLP.
        output (nn.Linear): Callable that applies a linear transformation to the output of the decoder.

    Implementation Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        token_embeddings: nn.Embedding,
        layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module,
        output: nn.Linear,
    ) -> None:
        super().__init__()
        self.tok_embeddings = token_embeddings

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(layer)

        self.norm = norm
        self.output = output

    def forward(self, tokens: Tensor, mask: Optional[Tensor] = None, curr_pos: int = 0) -> Tensor:
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

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask, curr_pos)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = self.output(h).float()
        return output
