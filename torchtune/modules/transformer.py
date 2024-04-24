# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import copy
from typing import Optional

import torch
from torch import nn, Tensor

from torchtune.modules import CausalSelfAttention, KVCache


class TransformerDecoderLayer(nn.Module):
    """Transformer layer derived from the Llama2 model. Normalization is applied before the attention **and** FF layer.

    Args:
        attn (CausalSelfAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (nn.Module): Normalization to be applied before self-attention.
        mlp_norm (nn.Module): Normalization to be applied before the feed-forward layer.
    """

    def __init__(
        self,
        attn: CausalSelfAttention,
        mlp: nn.Module,
        sa_norm: nn.Module,
        mlp_norm: nn.Module,
    ) -> None:
        super().__init__()
        self.sa_norm = sa_norm
        self.attn = attn
        self.mlp_norm = mlp_norm
        self.mlp = mlp

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[Tensor]): Optional tensor which contains the mask.
                Only used during inference. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - d: embed dim

        TODO:
            - Make position of norm configurable
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied before self-attention
        attn_out = self.attn(self.sa_norm(x), mask, input_pos)

        # Residual connection; shape: [b, s, d]
        h = attn_out + x

        # Norm applied before the feedforward layer
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [b, s, d]
        out = h + mlp_out
        return out


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Return a list of ``n`` identical layers.

    Args:
        module (nn.Module): module to be cloned
        n (int): number of clones

    Returns:
        nn.ModuleList: list of ``n`` identical layers
    """
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Llama2 architecture.

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space.
        layer (TransformerDecoderLayer): Transformer Decoder layer.
        num_layers (int): Number of Transformer Decoder layers.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value. This is used to setup the
            :func:`~torchtune.modules.KVCache`
        head_dim (int): embedding dimension for each head in self-attention. This is used
            to setup the :func:`~torchtune.modules.KVCache`
        norm (nn.Module): Callable that applies normalization to the output of the decoder,
            before final MLP.
        output (nn.Module): Callable that applies a linear transformation to the output of
            the decoder.

    Note:
        Arg values are checked for correctness (eg: ``attn_dropout`` belongs to [0,1])
        in the module where they are used. This helps reduces the number of raise
        statements in code and improves readability.
    """

    def __init__(
        self,
        tok_embeddings: nn.Embedding,
        layer: TransformerDecoderLayer,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        norm: nn.Module,
        output: nn.Module,
    ) -> None:
        super().__init__()

        self.tok_embeddings = tok_embeddings
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.output = output
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype) -> None:
        for layer in self.layers:
            layer.attn.kv_cache = KVCache(
                max_batch_size=max_batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )

        # causal_mask is used during inference to ensure we're attending
        # to the right tokens
        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)
        )

    def forward(self, tokens: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape [b x s]
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        mask = None
        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, None, input_pos]

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask, input_pos)

        # shape: [b, s, d]
        h = self.norm(h)

        output = self.output(h).float()
        return output
