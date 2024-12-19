# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class VectorQuantizedEmbeddings(nn.Module):
    """
    Vector quantized embedding layer that takes in the output of an encoder
    and performs a nearest-neighbor lookup in the embedding space.
    Vector quantization was introduced in Oord et al. 2017 (https://arxiv.org/pdf/1711.00937.pdf)
    to generate high-fidelity images, videos, and audio data.

    This module currently does not support pre-training of the embeddings via EMA.

    Code was adapted from torchmultimodal's `Codebook module
    <https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/codebook.py>`_.

    Args:
        num_embeddings (int): Number of vectors in the embedding space.
        embedding_dim (int): Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z (Tensor): Tensor containing a batch of encoder outputs of shape ``(b, s, d)``, where
                b is batch size, s is sequence length or time, and d is ``embedding_dim``.

        Returns:
            Tuple[Tensor, Tensor]: The quantized input and the embedding vector ids that were used.

        Raises:
            ValueError: if input embedding dimension does not match embedding dimension of module
        """
        bsz, seq_len, z_embed_dim = z.shape
        if z_embed_dim != self.embedding_dim:
            raise ValueError(
                f"Expected last dimension of input tensor ({z_embed_dim}) to be embedding size of {self.embedding_dim}"
            )

        # Flatten into batch dimension
        z_flat = z.view(-1, z_embed_dim)
        # Calculate distances from each encoder, E(x), output vector to each embedding vector, e, ||E(x) - e||^2
        distances = torch.cdist(z_flat, self.embedding, p=2.0) ** 2

        # Encoding - select closest embedding vectors, shape [b * s, ]
        token_ids_flat = torch.argmin(distances, dim=1)

        # Quantize - shape [b * s, d]
        quantized_flat = self.decode(token_ids_flat)

        # Straight through estimator
        quantized_flat = z_flat + (quantized_flat - z_flat).detach()

        # Reshape to original - [b, s, d] and [b, s]
        quantized = quantized_flat.view(bsz, seq_len, z_embed_dim)
        token_ids = token_ids_flat.view(bsz, seq_len)

        return quantized, token_ids

    def extra_repr(self) -> str:
        return "num_embeddings={}, embedding_dim={}".format(
            self.num_embeddings, self.embedding_dim
        )

    def decode(self, token_ids: Tensor) -> Tensor:
        # Returns the embeddings of shape [b, s, d]
        return F.embedding(token_ids, self.embedding)
