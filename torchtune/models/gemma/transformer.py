# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchtune.modules import KVCache

from torchtune.modules.transformer import _get_clones, TransformerDecoderLayer


class GemmaTransformerDecoder(nn.Module):
    """
    Transformer Decoder derived from the Gemma architecture. A key difference between
    the Gemma transformer decoder and :class:`~torchtune.modules.TransformerDecoder`
    is that the output projection is replaced instead with a reverse projection
    using the transposed token embedding weights from output dim to input dim
    (see https://github.com/keras-team/keras-nlp/blob/master/keras_nlp/layers/modeling/reversible_embedding.py#L21).

    Args:
        tok_embeddings (nn.Embedding): PyTorch embedding layer, to be used to move
            tokens to an embedding space and as the output projection.
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
        norm_embeddings (bool): Whether to normalize the embeddings before passing them
            through the decoder layers. Defaults to False.

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
        norm_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.tok_embeddings = tok_embeddings
        self.layers = _get_clones(layer, num_layers)
        self.norm = norm
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.causal_mask = None
        self.norm_embeddings = norm_embeddings

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

        Returns:
            Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

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

        mask = None
        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, None, input_pos]

        if self.norm_embeddings:
            hidden_dim = h.size(-1)
            h = h * torch.tensor(hidden_dim**0.5, dtype=h.dtype)

        for layer in self.layers:
            # shape: [b, s, d]
            h = layer(h, mask, None)

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = F.linear(h, self.tok_embeddings.weight).float()
        return output
