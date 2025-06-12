import torch
import torch.nn as nn
from typing import Any, Optional


class Qwen2_5_VLRotaryPositionalEmbeddings(nn.Module):
    """
    This class implements two-dimensional Rotary Positional Embeddings (RoPE) for images
    based on the axial frequency 2D RoPE described in https://arxiv.org/pdf/2403.13298.

    The position embedding is simply applied to the x-axis and y-axis separately, encoding
    the x and y position of each patch within every tile.. The embedding is applied to each
    tile identically.

    Note: This module assumes the CLS token embedding is appended at the end of the sequence.

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the full input image. In this case, the function will consider your image as a single tile.
        dim (int): Embedding dimension. Unlike :class:`~torchtune.modules.RotaryPositionalEmbeddings`, this is
            usually set to the dim of each head in the attention module divided by 2, computed as
            ``embed_dim // num_heads // 2``. The divide by 2 accounts for x and y positions.
        base (int): The base for the geometric progression used to compute
            the rotation angles
        append_cls_token (bool): Set to True if CLS token embedding is at the end of the sequence in the vision transformer,
            False if is in the beginning of the sequence. RoPE is zeroed out for the CLS token. Default is True.
    """

    def __init__(
        self,
        patch_size: int,
        tile_size: int,
        dim: int,
        base: int = 10_000,
        append_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.patch_grid_size = tile_size // patch_size
        self.seq_len = self.patch_grid_size**2 + 1
        self.dim = dim
        self.base = base
        self.append_cls_token = append_cls_token
        self.rope_init()

    def rope_init(self):
        dim = self.dim // 2
        theta = 1.0 / (
            self.base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        # TODO replace with proper indicies
        # Create position indices for each patch in the tile
        patches_per_tile = self.patch_grid_size**2
        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )        
        patch_idx = torch.cat(
            [
                -1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device),
                patch_idx,
            ]
        )
        # Encode x and y positions of each patch in the tile
        patch_x_pos = patch_idx % self.patch_grid_size
        patch_y_pos = patch_idx // self.patch_grid_size

        # Outer product of theta and position index; output tensor has
        # a shape of [patches_per_tile + 1, dim // 4]
        x_theta = torch.einsum("i, j -> ij", patch_x_pos + 1, self.theta).float()
        y_theta = torch.einsum("i, j -> ij", patch_y_pos + 1, self.theta).float()

        # Shape: [patches_per_tile + 1, dim]
        freqs = torch.cat([x_theta, y_theta], dim=-1)
        # Zero out CLS token position frequencies
        freqs = freqs.masked_fill(patch_idx.unsqueeze(-1) < 0, 0)

        # cache includes both the cos and sin components and so the output shape is
        # [patches_per_tile + 1, dim, 2]
        cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def get_pos_indices(self):
        pass

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
            **kwargs (Any): additional keyword arguments. This is kept to match the forward signature of
                :class:`~torchtune.modules.RotaryPositionalEmbeddings`.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        bsz, _, n_h, h_d = x.shape

        # reshape input; the last dimension is used for computing the output.
        # Split tile dimension from the sequence dimension
        # Cast to float to match the reference implementation
        # tensor has shape [b, max_num_tiles, s // max_num_tiles, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(bsz, -1, self.seq_len, n_h, h_d // 2, 2)

        # reshape the cache for broadcasting
        rope_cache = self.cache.view(1, 1, self.seq_len, 1, h_d // 2, 2)

        # tensor has shape [b, max_num_tiles, s // max_num_tiles, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # Squash tile dimension back into sequence dimension - tensor has shape [b, s, n_h, h_d]
        x_out = x_out.reshape(bsz, -1, n_h, h_d)
        return x_out.type_as(x)
    
# TODO: make MultiModalPositionalEmbeddings for the decoder