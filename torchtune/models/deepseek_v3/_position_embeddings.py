# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from torch import nn


class DeepSeekV3YarnRotaryEmbeddings(nn.Module):
    """
    This class implements YaRN (Yet another RoPE extensioN) Rotary Positional Embeddings
    for DeepSeek v3, proposed in https://arxiv.org/abs/2309.00071.

    YaRN extends RoPE to longer sequence lengths by selectively applying frequency scaling
    to different parts of the frequency spectrum based on wavelength characteristics.
    It also includes magnitude scaling to preserve attention patterns.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scaling_factor (float): Factor by which to scale the original context length
        original_max_seq_len (int): Original maximum sequence length before scaling
        beta_fast (float): Lower bound for frequency scaling range. Default: 32
        beta_slow (float): Upper bound for frequency scaling range. Default: 1
        mscale (float): Magnitude scaling factor. Default: 1
        mscale_all_dim (float): Magnitude scaling for all dimensions. Default: 0
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        scaling_factor: float = 1.0,
        original_max_seq_len: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.scaling_factor = scaling_factor
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.rope_init()

    def _find_correction_dim(
        self, num_rotations: float, dim: int, base: int, max_position_embeddings: int
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _find_correction_range(
        self, low_rot: float, high_rot: float, dim: int, base: int, max_position_embeddings: int
    ) -> tuple[int, int]:
        low = math.floor(
            self._find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    def get_mscale(self, scale: float = 1.0, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _get_linear_ramp_mask(self, min_val: int, max_val: int, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001

        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def rope_init(self):
        # Compute base extrapolated freqs
        freq_base = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # Compute scaled interpolated freqs
        freq_interp = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # Find correction range for frequency interpolation
        low, high = self._find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_seq_len,
        )

        # Create interpolation mask
        inv_freq_mask = 1.0 - self._get_linear_ramp_mask(low, high, self.dim // 2)

        # Interpolate between scaled and unscaled frequencies
        theta = freq_interp * (1 - inv_freq_mask) + freq_base * inv_freq_mask

        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # Calculate magnitude scaling
        mscale = float(
            self.get_mscale(self.scaling_factor, self.mscale)
            / self.get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([idx_theta.cos() * mscale, idx_theta.sin() * mscale], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
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
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
