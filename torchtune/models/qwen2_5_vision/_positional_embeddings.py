# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional, Tuple


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
