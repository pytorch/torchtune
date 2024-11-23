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
    The embedding weights are trained with exponential moving average updates as described
    in original paper.

    Code was adapted from torchmultimodal's `Codebook module
    <https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/layers/codebook.py>`_.

    Args:
        num_embeddings (int): Number of vectors in the embedding space.
        embedding_dim (int): Dimensionality of the embedding vectors.
        decay (float, optional): Factor used in exponential moving average update of the embeddings.
            Defaults to ``0.99``.
        codebook_usage_threshold (float, optional): Threshold for the average number of times an embedding vector
            is chosen below which it will be re-initialized. Defaults to ``1.0``.
        learnable (bool): If True, register embedding weights, codebook usage, and codebook average to buffer
            for EMA updates during training. If False, only register embedding weights as an nn.Parameter, for use
            in a frozen module. Default is False.
        epsilon (float, optional): Noise used in Laplace smoothing of codebook usage. Defaults to ``1e-7``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        codebook_usage_threshold: float = 1.0,
        learnable: bool = False,
        epsilon: float = 1e-7,
    ) -> None:
        super().__init__()
        # Embedding weights and parameters for EMA update will be registered to buffer, as they
        # will not be updated by the optimizer but are still model parameters.
        # code_usage and code_avg correspond with N and m, respectively, from Oord et al.
        randn_init_embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embedding", randn_init_embedding.clone())
        if learnable:
            self.register_buffer("code_usage", torch.zeros(num_embeddings))
            self.register_buffer("code_avg", randn_init_embedding.clone())

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.learnable = learnable

        self._decay = decay
        # Used in Laplace smoothing of code usage
        self._epsilon = epsilon

        # Threshold for randomly reseting unused embedding vectors
        self.codebook_usage_threshold = codebook_usage_threshold

    def _tile(self, x: Tensor, n: int) -> Tensor:
        # Repeat vectors in x if x has less than n vectors
        num_vectors, num_channels = x.shape
        if num_vectors < n:
            num_repeats = (n + num_vectors - 1) // num_vectors
            # Add a small amount of noise to repeated vectors
            std = 0.01 / torch.sqrt(torch.tensor(num_channels))
            x = x.repeat(num_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _get_random_vectors(self, x: Tensor, n: int) -> Tensor:
        # Gets n random row vectors from 2D tensor x
        x_tiled = self._tile(x, n)
        idx = torch.randperm(x_tiled.shape[0])
        x_rand = x_tiled[idx][:n]
        return x_rand

    def _ema_update_embedding(self, z: Tensor, codebook_indices: Tensor) -> None:
        # Closed form solution of codebook loss, ||e - E(x)||^2, is simply the average
        # of the encoder output. However, we can't compute this in minibatches, so we
        # must use exponential moving average.

        # Convert indices to one hot encoding
        codebook_onehot = nn.functional.one_hot(
            codebook_indices, num_classes=self.num_embeddings
        ).type(torch.float)
        # Count how often each embedding vector was looked up
        codebook_selection_count = torch.sum(codebook_onehot, 0)
        # Update usage value for each embedding vector
        self.code_usage.mul_(self._decay).add_(
            codebook_selection_count, alpha=(1 - self._decay)
        )
        # Laplace smoothing of codebook usage - to prevent zero counts
        n = torch.sum(self.code_usage)
        self.code_usage.add_(self._epsilon).divide_(
            n + self.num_embeddings * self._epsilon
        ).mul_(n)
        # Get all encoded vectors attracted to each embedding vector
        encoded_per_codebook = torch.matmul(codebook_onehot.t(), z)
        # Update each embedding vector with new encoded vectors that are attracted to it,
        # divided by its usage to yield the mean of encoded vectors that choose it
        self.code_avg.mul_(self._decay).add_(
            encoded_per_codebook, alpha=(1 - self._decay)
        )
        self.embedding = self.code_avg / self.code_usage.unsqueeze(1)
        # Reset any embedding vectors that fall below threshold usage with random encoded vectors
        z_rand = self._get_random_vectors(z, self.num_embeddings)
        self.embedding = torch.where(
            self.code_usage.unsqueeze(1) >= self.codebook_usage_threshold,
            self.embedding,
            z_rand,
        )

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
        quantized_flat = self.lookup(token_ids_flat)

        # Use exponential moving average to update the embedding instead of a codebook loss,
        # as suggested by Oord et al. 2017 and Razavi et al. 2019.
        if self.training and self.learnable:
            self._ema_update_embedding(z_flat, token_ids_flat)

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

    def lookup(self, token_ids: Tensor) -> Tensor:
        # Returns the embeddings of shape [b, s, d]
        return F.embedding(token_ids, self.embedding)
