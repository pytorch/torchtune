# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in: https://arxiv.org/abs/2104.09864

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450

    Attributes:
        dim (int): Embedding dimension for each head, computed as:
            embed_size //  num_heads
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles

    Args:
        x (tensor): input tensor to which rope is applied

    Returns:
        torch.Tensor: output tensor with RoPE applied

    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim

        theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("theta", theta)
        self.build_rope_cache(max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: The implementation below can be made more efficient
        for inference.
        """
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len]

        # reshape input; the last dimension is used for computing the output
        # cast to float to match the reference implementation
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out2 = x_out2.flatten(3)
        return x_out2.type_as(x)
