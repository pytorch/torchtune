# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen25VLRotaryPositionalEmbeddings(nn.Module):
    """
    M-RoPE (Multimodal Rotary Embeddings) for Qwen2.5-VL.

    Initially described in https://arxiv.org/pdf/2409.12191.

    Extends standard 1D RoPE to three axes: time, height, width.

    Unlike the huggingface implementation, this implementation caches the RoPE tables
    for each position and each of the three dimensions.
        Args:
        head_dim (int):      dimensionality per head (e.g. 128)
        max_seq_len (int):   maximum temporal length to expect (default 128000)
        max_height (int):    maximum height to expect (default 4096)
        max_width (int):     maximum width to expect (default 4096)
        base (float):        geometric base for theta (default 1e6)
        mrope_section (list[int]): number of frequency-pairs for [time, height, width] (default [16, 24, 24])
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 128000,
        max_height: int = 4096,
        max_width: int = 4096,
        base: float = 1000000.0,
        mrope_section: Optional[list[int]] = None,
    ) -> None:
        super().__init__()

        if mrope_section is None:
            mrope_section = [16, 24, 24]

        if sum(mrope_section) * 2 != head_dim:
            raise ValueError(
                f"mrope_section pairs {mrope_section} must satisfy 2*sum = head_dim ({head_dim})"
            )

        self.head_dim = head_dim

        self.max_seq_len = max_seq_len
        self.max_height = max_height
        self.max_width = max_width

        self.base = base
        self.mrope_section = mrope_section

        self.rope_init()

    def rope_init(self) -> None:
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        attention_scaling = 1.0
        self.register_buffer("theta", theta, persistent=False)
        self.attention_scaling = attention_scaling

        self.build_rope_cache("time", self.max_seq_len)
        self.build_rope_cache("height", self.max_height)
        self.build_rope_cache("width", self.max_width)

    def build_rope_cache(self, name: str, length: int):
        # positions 0…length-1
        p = torch.arange(length, device=self.theta.device, dtype=self.theta.dtype)
        # [length, head_dim/2]
        angles = torch.einsum("p,f->pf", p, self.theta).float()
        # [length, head_dim]
        freqs = torch.cat([angles, angles], dim=-1)
        # [length, 2*head_dim]
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        self.register_buffer(f"{name}_cache", cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        input_pos: torch.LongTensor,
        *,
        window_index: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute M-RoPE cos/sin tables for a batch of queries/keys.

        Args:
            x (torch.Tensor): input tensor with shape ``[B, s_x, n_heads, head_dim]``
            input_pos (torch.LongTensor): the time, height, width indices with shape ``[3, B, L]``
            window_index (Optional[torch.Tensor]): Optional tensor for window indexing (not used in M-RoPE)

        Returns:
            q_out (torch.Tensor): output tensor with shape ``[B, s_x, n_heads, head_dim]``

        Notation used for tensor shapes:
            - B: batch size
            - s_x: sequence length
            - n_heads: number of attention heads
            - head_dim: dimension of each head
            - L: sequence length
            - D: head dimension
        """
        sections = self.mrope_section * 2

        # unpack input_pos into three tensors of shape [B, L]
        t_ids, h_ids, w_ids = input_pos

        # retrieve caches at position index, returns tensor of shape []
        cache_t = self.time_cache[t_ids]
        cache_h = self.height_cache[h_ids]
        cache_w = self.width_cache[w_ids]

        # [3, B, L, 2*D]
        stacked = torch.stack([cache_t, cache_h, cache_w], dim=0)

        cos3 = stacked[..., : self.head_dim] * self.attention_scaling
        sin3 = stacked[..., self.head_dim :] * self.attention_scaling

        # split into chunks of size self.mrope_section
        cos_chunks = cos3.split(sections, dim=-1)
        sin_chunks = sin3.split(sections, dim=-1)

        # for each block, pick the modality slice
        cos_parts = [cos_chunks[i][i % 3] for i in range(len(cos_chunks))]
        sin_parts = [sin_chunks[i][i % 3] for i in range(len(sin_chunks))]

        # concat back to [B, L, D] and unsqueeze heads-axis → [B,1,L,D]
        # NOTE: the head dimension is the axis 2
        cos = torch.cat(cos_parts, dim=-1).unsqueeze(2)
        sin = torch.cat(sin_parts, dim=-1).unsqueeze(2)

        x_out = (x * cos) + (rotate_half(x) * sin)
        return x_out.to(x.dtype)


class Qwen25VisionRotaryPositionalEmbeddings(nn.Module):
    """
    2D Rope for Qwen 2.5 VL's Vision Transformer

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        spatial_merge_unit (int): size of a spatial merge unit,
            aka the number of patches that share the same position index
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        spatial_merge_unit: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.spatial_merge_unit = spatial_merge_unit
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
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

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_pos: Optional[torch.Tensor] = None,
        window_index: Optional[torch.Tensor] = None,
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
            window_index (Optional[torch.Tensor]): Optional tensor which contains the window index
                of each token. During training, this is used to indicate the window index
                of each token when packed, shape [b, s].

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
        # merge height and width into one dimension
        rope_cache = rope_cache.flatten(1)  # [s, h_d]

        # rearrange indices to match window index
        rope_cache = rope_cache.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rope_cache = rope_cache[window_index, :, :]
        rope_cache = rope_cache.reshape(seq_len, -1)

        # reshape input; the last dimension is used for computing the output.
        x_float = x.float()
        half_dim = x_float.shape[-1] // 2
        x1 = x_float[..., :half_dim]
        x2 = x_float[..., half_dim:]
        xshaped = torch.stack([x1, x2], dim=-1)

        # reshape the cache for broadcasting
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

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
