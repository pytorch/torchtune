# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
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


class VisionRotaryPositionalEmbeddings(nn.Module):
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
        # Create position indices for each patch in the tile
        patches_per_tile = self.patch_grid_size**2
        patch_idx = torch.arange(
            patches_per_tile, dtype=self.theta.dtype, device=self.theta.device
        )
        # Add a placeholder index for CLS token - will not be used in RoPE
        if self.append_cls_token:
            patch_idx = torch.cat(
                [
                    patch_idx,
                    -1 * torch.ones(1, dtype=patch_idx.dtype, device=patch_idx.device),
                ]
            )
        else:
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

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
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
