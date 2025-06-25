# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import torch
from torch import nn


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
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
        self.spatial_merge_unit = spatial_merge_unit # TODO: should this be an attr or just merge size
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
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None, window_index: Optional[torch.Tensor] = None
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
                of each token when packed, shape [b, s]. # TODO: check if this is correct


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
        rope_cache = rope_cache.flatten(1) # [s, h_d, 2]

        # rearrange indices to match window index
        rope_cache = rope_cache.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rope_cache = rope_cache[window_index, :, :]
        rope_cache = rope_cache.reshape(seq_len, -1)

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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)




class Qwen25VLRotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Multimodal Rotary Positional Embeddings (MRoPE) for Qwen2.5-VL.

    MRoPE extends standard RoPE to handle 3D position embeddings:
    - Temporal dimension (for videos)
    - Height dimension (spatial)
    - Width dimension (spatial)

    For text-only tokens, all three dimensions use the same position IDs, making it
    equivalent to standard 1D RoPE. The key innovation is that different parts of
    the embedding dimension handle different spatial dimensions.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        mrope_section (List[int]): The dimensions allocated to temporal, height, and width.
            Should sum to head_dim. Default: [16, 24, 24] (sum=64 for typical head_dim)
        max_seq_len (int): Maximum expected sequence length for the model, if exceeded
            the cached freqs will be recomputed. Default: 32768
        base (float): The base for the geometric progression used to compute
            the rotation angles. Default: 1000000.0
    """

    def __init__(
        self,
        dim: int,
        mrope_section: List[int],
        max_seq_len: int = 32768,
        base: float = 1000000.0,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        # In HuggingFace implementation, mrope_section is doubled for the full head dimension
        # [16, 24, 24] becomes [16, 24, 24, 16, 24, 24] which sums to 128 for head_dim=128
        self.mrope_section = mrope_section * 2
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        # Compute theta for each section separately
        # Temporal section
        temporal_dim = self.mrope_section[0]
        temporal_theta = 1.0 / (
            self.base ** (torch.arange(0, temporal_dim, 2).float() / temporal_dim)
        )
        
        # Height section  
        height_dim = self.mrope_section[1]
        height_theta = 1.0 / (
            self.base ** (torch.arange(0, height_dim, 2).float() / height_dim)
        )
        
        # Width section
        width_dim = self.mrope_section[2]
        width_theta = 1.0 / (
            self.base ** (torch.arange(0, width_dim, 2).float() / width_dim)
        )
        
        self.register_buffer("temporal_theta", temporal_theta, persistent=False)
        self.register_buffer("height_theta", height_theta, persistent=False)
        self.register_buffer("width_theta", width_theta, persistent=False)
        
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 32768) -> None:
        # Create position indexes for each dimension
        seq_idx = torch.arange(max_seq_len, dtype=self.temporal_theta.dtype, device=self.temporal_theta.device)

        # Compute frequency matrices for each dimension
        temporal_freqs = torch.outer(seq_idx, self.temporal_theta).float()
        height_freqs = torch.outer(seq_idx, self.height_theta).float()
        width_freqs = torch.outer(seq_idx, self.width_theta).float()

        # Cache includes both cos and sin components for each dimension
        # Shape: [max_seq_len, dim_section//2, 2]
        temporal_cache = torch.stack([torch.cos(temporal_freqs), torch.sin(temporal_freqs)], dim=-1)
        height_cache = torch.stack([torch.cos(height_freqs), torch.sin(height_freqs)], dim=-1) 
        width_cache = torch.stack([torch.cos(width_freqs), torch.sin(width_freqs)], dim=-1)
        
        self.register_buffer("temporal_cache", temporal_cache, persistent=False)
        self.register_buffer("height_cache", height_cache, persistent=False)
        self.register_buffer("width_cache", width_cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids.
                Can be either:
                - 2D tensor with shape [b, s] for standard RoPE (will be expanded to 3D)
                - 3D tensor with shape [3, b, s] for MRoPE where 3 represents [temporal, height, width]
                If None, assume the index of the token is its position id. Default is None.

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

        if input_pos is None:
            # Create default sequential positions for all dimensions
            device = x.device
            pos_1d = torch.arange(seq_len, device=device)
            input_pos = pos_1d.unsqueeze(0).expand(3, 1, -1)  # [3, 1, s]
            input_pos = input_pos.expand(3, x.size(0), -1)    # [3, b, s]
        elif input_pos.dim() == 2:  # [b, s]
            # Convert 2D to 3D by replicating across all 3 dimensions
            input_pos = input_pos.unsqueeze(0).expand(3, -1, -1)  # [3, b, s]

        # Extract position indices for each dimension
        temporal_pos = input_pos[0]  # [b, s]
        height_pos = input_pos[1]    # [b, s]  
        width_pos = input_pos[2]     # [b, s]

        # Extract cached values for each dimension
        temporal_rope = self.temporal_cache[temporal_pos]  # [b, s, temporal_dim//2, 2]
        height_rope = self.height_cache[height_pos]        # [b, s, height_dim//2, 2]
        width_rope = self.width_cache[width_pos]           # [b, s, width_dim//2, 2]

        # Apply rotations for each section of the embedding
        return self._apply_mrope_rotation(x, temporal_rope, height_rope, width_rope)

    def _apply_mrope_rotation(
        self, 
        x: torch.Tensor, 
        temporal_rope: torch.Tensor, 
        height_rope: torch.Tensor, 
        width_rope: torch.Tensor
    ) -> torch.Tensor:
        """Apply MRoPE rotation to different sections of the embedding dimension."""
        b, s, n_h, h_d = x.shape
        
        # The mrope_section is doubled: [16, 24, 24, 16, 24, 24]
        # We need to split into 6 sections and apply rotations in pairs
        temporal_dim = self.mrope_section[0]  # 16
        height_dim = self.mrope_section[1]    # 24  
        width_dim = self.mrope_section[2]     # 24
        
        # Split into 6 sections
        sections = []
        start_idx = 0
        for dim in self.mrope_section:
            sections.append(x[..., start_idx:start_idx+dim])
            start_idx += dim
            
        # Apply rotations to corresponding pairs
        # Sections 0,3 get temporal rotation; 1,4 get height; 2,5 get width
        rotated_sections = []
        for i, section in enumerate(sections):
            rope_cache = [temporal_rope, height_rope, width_rope][i % 3]
            rotated_sections.append(self._apply_rotation_to_section(section, rope_cache))

        # Concatenate all rotated sections back together
        x_out = torch.cat(rotated_sections, dim=-1)
        return x_out

    def _apply_rotation_to_section(self, x_section: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        """Apply rotation to a specific section of the embedding."""
        # x_section: [b, s, n_h, section_dim]
        # rope_cache: [b, s, section_dim//2, 2]
        
        # Reshape input for rotation: [b, s, n_h, section_dim//2, 2]
        x_shaped = x_section.float().reshape(*x_section.shape[:-1], -1, 2)
        
        # Reshape cache for broadcasting: [b, s, 1, section_dim//2, 2]
        rope_cache = rope_cache.unsqueeze(2)
        
        # Apply rotation
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        
        # Flatten back to original shape
        x_out = x_out.flatten(-2)
        return x_out.type_as(x_section)
