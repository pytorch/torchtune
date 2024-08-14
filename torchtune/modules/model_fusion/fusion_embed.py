# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch import nn, Tensor


class FusionEmbedding(nn.Module):
    """Fusion embedding supports training additional special tokens while keeping
    the original embedding frozen. When fusing new models with a langauge model,
    there may be some additional tokens needed to support the fused langauge model.
    The FusionEmbedding keeps the original embeddings frozen while learning a much smaller
    second embedding for the additional tokens. During forward this module routes
    the tokens to the appropriate embedding table.

    Args:
        vocab_size (int): language model vocab size
        additional_tokens (int): additional tokens for the fused model
        embed_dim (int): embedding dimension of the two embedding tables
    """

    def __init__(self, vocab_size: int, additional_tokens: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(additional_tokens, embed_dim)
        self.dim = embed_dim
        # TODO: Support merging the embeddings after finetuning

    def fusion_params(self) -> List[str]:
        """
        Return fusion embedding parameters.
        """
        fusion_params = ["fusion_embedding.weight"]
        return fusion_params

    def _fused_embed(self, bs, seq_len):
        """
        Return an empty tensor the shape of the combined embedding.
        """
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        return torch.empty(bs, seq_len, self.dim, device=device, dtype=dtype)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): input integer tensor with shape
                [batch_size x seq_length]

        Returns:
            Tensor: output tensor embedding with shape
                [batch_size x seq_length x embed_dim]`

        """
        bs, seq_len = input.size()
        vocab_size = self.embedding.num_embeddings

        mask = input < vocab_size
        tokens = torch.masked_select(input, mask)
        additional_tokens = torch.masked_select(input, ~mask) - vocab_size

        embeds = self.embedding(tokens)
        additional_embeds = self.fusion_embedding(additional_tokens)

        out = self._fused_embed(bs, seq_len)
        mask = mask.unsqueeze(-1).expand(bs, seq_len, self.dim)
        out.masked_scatter_(mask, embeds)
        out.masked_scatter_(~mask, additional_embeds)
        return out
