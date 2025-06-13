# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Callable

import torch
from torch import nn

from torchtune.modules import Fp32LayerNorm
from torchtune.modules.transformer import _get_clones
from torchtune.modules.model_fusion import register_fusion_module


class Qwen2_5_VisionRotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta
                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached = None

    def build_rope_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            # Double the cache size to avoid frequent rebuilds
            seqlen *= 2
            self._seq_len_cached = seqlen
            # Recompute inv_freq to ensure it's on the right device
            self.inv_freq = 1.0 / (self.theta ** (torch.arange(
                0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device)
                                                / self.dim))
            seq = torch.arange(seqlen,
                               device=self.inv_freq.device,
                               dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Get rotary position embeddings for given sequence length.
        
        Args:
            seqlen (int): Sequence length
            
        Returns:
            torch.Tensor: Frequencies tensor of shape [seqlen, dim//2]
        """
        self.build_rope_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2_5_VisionMLP(nn.Module):
    """
    MLP for Qwen 2.5 Vision.
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.down_proj = down_proj
        self.up_proj = up_proj
        self.act_fn = activation

    def forward(self, x: torch.Tensor):
        x_gate, _ = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up, _ = self.up_proj(x)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down

class Qwen2_5_VisionTransformer(nn.Module):
    """
    Vision transformer for Qwen 2.5 VL that processes images and videos using grid-based processing.
    Matches HF's implementation while maintaining torchtune's performance optimizations.
    """

    def __init__(
        self,
        patch_size: int,
        num_layers: int,
        embed_dim: int,
        layer: nn.Module,
        token_pos_embedding: nn.Module,
        full_att_block_indexes: List[int],
        pre_tile_pos_embed: Optional[nn.Module] = None,
        post_tile_pos_embed: Optional[nn.Module] = None,
        in_channels: int = 3,
        spatial_merge_size: int = 2,
        window_size: int = 14,
        temporal_patch_size: int = 2,
    ) -> None:
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if full_att_block_indexes and (len(full_att_block_indexes) > num_layers):
            raise ValueError(
                f"len(out_indices) must be <= num_layers. Got {full_att_block_indexes=} and {num_layers=}"
            )

        # Spatial merging configuration
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.temporal_patch_size = temporal_patch_size

        self.full_att_block_indexes = full_att_block_indexes
        self.embed_dim = embed_dim # TODO: remove if not used

        # input modules
        self.layers = _get_clones(layer, num_layers)
        self.token_pos_embedding = token_pos_embedding

        # Initialize rotary position embeddings
        # For vision transformers, we typically use head_dim // 2 for 2D positioning
        head_dim = embed_dim // 16  # Assuming 16 heads as default, should be parameterized
        # TODO: remove exact module for generalization
        self.rotary_pos_emb = Qwen2_5_VisionRotaryPositionalEmbeddings(dim=head_dim // 2)

        # 3D Convolution for patch embedding with temporal support - matches HF implementation
        # Following torchtune's pattern of keeping conv inside the transformer
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )


    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Calculate rotary position embeddings for spatial grid positions.
        
        This method computes position IDs for height and width dimensions,
        taking into account spatial merging for efficient processing.
        Matches HF's implementation exactly while using torchtune's caching.
        
        Args:
            grid_thw (torch.Tensor): Tensor of shape [num_items, 3] containing
                temporal, height, and width dimensions for each item.
                
        Returns:
            torch.Tensor: Position embeddings tensor with shape [total_positions, embed_dim]
        """
        pos_ids = []
        
        for t, h, w in grid_thw:
            # Create height position IDs - matches HF exactly
            hpos_ids = torch.arange(h, device=grid_thw.device).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            # Create width position IDs - matches HF exactly
            wpos_ids = torch.arange(w, device=grid_thw.device).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            
            # Stack and repeat for temporal dimension - matches HF exactly
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        
        # Concatenate all position IDs - matches HF exactly
        pos_ids = torch.cat(pos_ids, dim=0)
        
        # Get maximum grid size for computing rotary embeddings
        max_grid_size = grid_thw[:, 1:].max()
        
        # Use torchtune's cached rotary embedding computation
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        
        # Index into the full rotary embeddings and flatten - matches HF exactly
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process flattened patches using grid-based processing.
        
        Args:
            pixel_values (torch.Tensor): Flattened patches tensor of shape 
                [bsz, num_patches, channels * temporal_patch_size * patch_size * patch_size]
            grid_thw (torch.Tensor): Grid dimensions tensor of shape [num_items, 3] containing
                temporal, height, and width dimensions for each item.
                
        Returns:
            torch.Tensor: Processed embeddings of shape [bsz, num_patches, embed_dim]
        """
        # Reshape flattened patches for 3D convolution
        # From [num_patches, channels * temporal_patch_size * patch_size * patch_size]
        # to [num_patches, channels, temporal_patch_size, patch_size, patch_size]
        bsz, num_patches, _ = pixel_values.shape
        channels = pixel_values.shape[1] // (self.temporal_patch_size * self.patch_size * self.patch_size)
        
        x = pixel_values.view(
            bsz,
            num_patches,
            channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size
        )
        
        # Apply 3D convolution
        target_dtype = self.conv.weight.dtype
        x = self.conv(x.to(dtype=target_dtype))
        
        # Reshape to [bsz, num_patches, embed_dim]
        x = x.view(-1, num_patches, self.embed_dim)
        
        # Calculate total sequence length from grid dimensions
        total_t = grid_thw[:, 0].sum()
        total_h = grid_thw[:, 1].sum() // self.spatial_merge_size
        total_w = grid_thw[:, 2].sum() // self.spatial_merge_size
        seq_len = total_t * total_h * total_w
        
        # Reshape to [bsz, seq_len, embed_dim]
        x = x[:, :seq_len]
        
        
        # Get rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        
        # Process through transformer layers
        for layer_idx, transformer_layer in enumerate(self.layers):
            if layer_idx in self.full_att_block_indexes:
                # TODO: Implement window attention logic
                pass
            x = transformer_layer(x, rotary_pos_emb)
        
        
        return x





class Qwen2_5VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, ``nn.Sequential(CLIP(), Qwen2_5VisionProjectionHead())``.

    Note: this module assumes the CLS token embedding is added at the end
    of the sequence.

    Args:
        output (nn.Module): output layer, typically an MLP.
        pixel_shuffle_scaling_factor (float): scaling factor for pixel shuffle.
    """

    def __init__(
        self,
        output: nn.Module,
    ) -> None:
        super().__init__()
        self.output = output


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b, e, d]

        Returns:
            Tensor: output tensor of a sequence of embeddings [b, s, d * pixel_shuffle_factor ** 2]

        Notation used for tensor shapes:
            - b: batch size
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by t * (e - 1) // (pixel_shuffle_factor ** 2)
            - d: embed dim
        """
        # Remove cls token - assumes it is the last token in the sequence
        x = x[:, :-1, :] # TODO: Remove?
        bsz, embeds, dim = x.shape

        h_patches = w_patches = int(embeds**0.5)
        x = x.reshape(bsz, h_patches, w_patches, -1)
        _, new_h_patches, new_w_patches, new_dim = x.shape
        # shape: [bsz, embeds // factor ** 2, dim * factor ** 2)]
        x = x.reshape(bsz, new_h_patches * new_w_patches, new_dim)
        # apply output - shape [bsz, embeds // factor ** 2, output_dim]
        x = self.output(x)

        return x



class Qwen2_5VisionEncoder(nn.Module):
    """Vision encoder model for Qwen 2.5. This combines a pretrained
    vision encoder with a learnable projection head. The projection head
    is converted to a fusion module and supports fusion utils.

    Args:
        visual_encoder (nn.Module): Qwen2_5_VisionTransformer model
        projection_head (nn.Module): ``projection_head`` that takes embeddings
            with dimension ``encoder_dim`` as input and outputs embeddings of
            size ``decoder_dim``. See :func:`torchtune.models.qwen2_5_vision.qwen2_5_vision_projection_head`
            as an example.
    """

    def __init__(self, visual_encoder: nn.Module, projection_head: nn.Module) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.projection = projection_head
        register_fusion_module(self.projection)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images (torch.Tensor): Image tensor with shape [b x c x w x h]

        Returns:
            Tensor: output tensor of a sequence of embeddings ``[b x s x d]``
                where sequence length (``s``) is ``(num_imgs*num_tiles)+num_embeds``

         Notation used for tensor shapes:
            - b: batch size, equal to flatten(batch x images x tiles)
            - c: number of image channels (e.g. rgb = 3)
            - w: image width
            - h: image height
            - s: sequence length computed by i*t*clip_embeds_per_tile
            - d: embed dim
        """
        #TODO: check dims
        x, _ = self.visual_encoder(images[:, None, None])
        x = self.projection(x.squeeze((1, 2)))
        return x

