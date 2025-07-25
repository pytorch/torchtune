# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch

from torch import nn


class Llama4ScaledRoPE(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864 with additional
    scaling from https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1.

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Note that this class is identical to :class:`~torchtune.models.llama4.Llama3ScaledRoPE`, but with different default values
    for scaling factors, as set for Llama4 Scout. See the meta-llama reference code here:

    https://github.com/meta-llama/llama-models/blob/28fa4e3b287e84f6a6a92aab3c931f7479c827c1/models/llama4/args.py#L100-L107

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scale_factor (Optional[float]): scaling factor for theta. Default: 16.0
        low_freq_factor (Optional[float]): low frequency factor for scaling theta. Default: 1.0
        high_freq_factor (Optional[float]): high frequency factor for scaling theta. Default: 1.0
        old_context_len (Optional[int]): old context length for scaling theta. Default: 8192
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 10485760,
        base: int = 500_000,
        scale_factor: Optional[float] = 16.0,
        low_freq_factor: Optional[float] = 1.0,
        high_freq_factor: Optional[float] = 1.0,
        old_context_len: Optional[int] = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        """
        Warning: this is called in recipes before torch.compile,
        so that the cache is built in advance.
        """
        freqs = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # If we're on meta device return early.
        # We can't apply scaling until freqs is filled with real data
        if freqs.is_meta:
            return

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def apply_scaling(
        self,
        freqs: torch.Tensor,
        scale_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        old_context_len: int,
    ):
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        Raises:
            RuntimeError: if RoPE cache is not initialized prior to forward call
        """

        if not self.is_cache_built:
            raise RuntimeError(
                "RoPE cache is not built. Please call rope_init() first."
            )

        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                torch.sub(
                    xshaped[..., 0] * rope_cache[..., 0],
                    xshaped[..., 1] * rope_cache[..., 1],
                ),
                torch.add(
                    xshaped[..., 1] * rope_cache[..., 0],
                    xshaped[..., 0] * rope_cache[..., 1],
                ),
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
