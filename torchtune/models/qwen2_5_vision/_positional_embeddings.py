# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Tuple


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

class Qwen2_5_VLRotaryEmbedding(nn.Module):
    """
    Multimodal Rotary Position Embedding for Qwen2.5-VL.
    
    This implements MRoPE which handles 3D position embeddings:
    - Temporal dimension (for videos)
    - Height dimension (spatial)  
    - Width dimension (spatial)
    
    For text tokens, all three dimensions use the same position IDs, making it
    equivalent to standard 1D RoPE.
    """
    
    def __init__(
        self,
        dim: int,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim (int): Dimension of the embedding (head_dim).
            base (float): Base for computing inverse frequencies.
            device (torch.device): Device to place tensors on.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Create inverse frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor (used for device/dtype inference).
            position_ids (torch.Tensor): Position IDs with shape (3, batch_size, seq_len)
                                        where 3 represents [temporal, height, width].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos, sin) embeddings with shape 
                                              (3, batch_size, seq_len, head_dim).
        """
        # Expand inv_freq to match position_ids structure
        # Shape: (3, batch_size, head_dim // 2, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        
        # Expand position_ids for matrix multiplication
        # Shape: (3, batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # Compute frequencies: (3, batch_size, head_dim // 2, seq_len)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            # Duplicate freqs for cos/sin pairs: (3, batch_size, seq_len, head_dim)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2_5_VLCompatibleRotaryEmbedding(nn.Module):
    """
    MultiHeadAttention-compatible version of Qwen2.5-VL's MRoPE.
    
    Stateless implementation that computes MRoPE on-the-fly from 3D position_ids.
    Works seamlessly with MultiHeadAttention's pos_embeddings interface.
    """
    
    def __init__(
        self,
        dim: int,
        mrope_section: list,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim (int): Dimension of the embedding (head_dim).
            mrope_section (list): Multimodal rope section [temporal_dim, height_dim, width_dim].
            base (float): Base for computing inverse frequencies.
            device (torch.device): Device to place tensors on.
        """
        super().__init__()
        self.dim = dim
        self.mrope_section = mrope_section
        
        # Create the underlying MRoPE module
        self.rope = Qwen2_5_VLRotaryEmbedding(dim, base, device)
    
    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor with shape [b, s, n_h, h_d] or [b, s, n_kv, h_d].
            input_pos (Optional[torch.Tensor]): Position IDs. If 3D with shape [3, b, s], 
                                              uses MRoPE. If 2D with shape [b, s], uses standard RoPE.
        
        Returns:
            torch.Tensor: Tensor with rotary embeddings applied.
        """
        if input_pos is None:
            return x
            
        # Handle 2D position_ids (fallback to standard RoPE behavior)
        if input_pos.dim() == 2:  # [b, s]
            # Convert to 3D by replicating across 3 dimensions
            input_pos = input_pos.unsqueeze(0).expand(3, -1, -1)
        
        # Compute cos/sin using the underlying MRoPE
        cos, sin = self.rope(x, input_pos)  # Both [3, b, s, h_d]
        
        # Apply mrope sectioning
        mrope_section = [s * 2 for s in self.mrope_section]  # Double for cos/sin pairs
        cos_parts = cos.split(mrope_section, dim=-1)
        sin_parts = sin.split(mrope_section, dim=-1)
        
        # Recombine sections: [cos_temporal, cos_height, cos_width, cos_temporal, ...]
        cos_sectioned = torch.cat([cos_parts[i % 3] for i in range(len(cos_parts))], dim=-1)
        sin_sectioned = torch.cat([sin_parts[i % 3] for i in range(len(sin_parts))], dim=-1)
        
        # Average over spatial dimensions and reshape for broadcasting
        cos_final = cos_sectioned.mean(0).unsqueeze(2)  # [b, s, 1, h_d]
        sin_final = sin_sectioned.mean(0).unsqueeze(2)  # [b, s, 1, h_d]
        
        # Apply rotation
        x_embed = (x * cos_final) + (rotate_half(x) * sin_final)
        return x_embed
