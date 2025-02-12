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
        max_num_tiles (int): The maximum number of tiles in the image. This is used to unfold the input sequence
            length into sequence length per tile so RoPE can be applied to each tile separately.
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
        max_num_tiles: int,
        dim: int,
        base: int = 10_000,
        append_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.patch_grid_size = tile_size // patch_size
        self.max_num_tiles = max_num_tiles
        self.dim = dim
        self.base = base
        self.append_cls_token = append_cls_token
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
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
        # a shape of [patches_per_tile + 1, dim // 2]
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

        Raises:
            ValueError: if sequence length of input tensor does not match the 2D RoPE cache's sequence length

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
        xshaped = x.float().reshape(bsz, self.max_num_tiles, -1, n_h, h_d // 2, 2)
        seq_len = xshaped.size(2)

        if seq_len != self.cache.shape[0]:
            raise ValueError(
                f"Input sequence length {seq_len} does not match 2D RoPE cache sequence length {self.cache.shape[0]}."
            )

        # reshape the cache for broadcasting
        rope_cache = self.cache.view(1, 1, seq_len, 1, h_d // 2, 2)

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
        x_out = x_out.reshape(bsz, self.max_num_tiles * seq_len, n_h, h_d)
        return x_out.type_as(x)


class FireSelfAttention(nn.Module):
    """
    This class implements FIRE (Functional Interpolation for Relative Positional Encodings)
    as described in https://arxiv.org/abs/2310.04418 for causal language modeling tasks. The
    only modification from the paper is that this implementation uses the GELU activation function instead
    of ReLU in order to avoid possible problems with "dying" neurons.

    This module is fundamentally a positional encoding scheme; however, due to the nature of FIRE relative
    positional encodings, it takes the form of an attention layer.

    Args:
        dim_model (int): The embedding dimension of the input vectors.
        num_heads (int): The number of self-attention heads, set to 1 by default. The dimension of each individual head
            is usually computed as ``dim_model // num_heads``.
        hidden_size (int): The dimension of the MLP layers in each attention head used to compute the bias matrix.

    Raises:
        ValueError: If num_heads does not divide dim_model
    """

    def __init__(
        self, dim_model: int, num_heads: int = 1, hidden_size: int = 32
    ) -> None:
        super().__init__()

        # make sure num_heads divides dim_model:
        if dim_model % num_heads != 0:
            raise ValueError("Number of heads must divide dimension of model")

        # compute kdim = vdim
        kdim = dim_model // num_heads

        # initialize attention heads
        self.attention_heads = nn.ModuleList(
            [
                self.FireAttentionHead(dim_model, kdim, hidden_size)
                for _ in range(num_heads)
            ]
        )

        # final linear layer
        self.W_o = nn.Linear(dim_model, dim_model, bias=False)

    class FireAttentionHead(nn.Module):
        """
        An inner class to implement a single attention head using the FIRE positional encoding scheme.
        **Do not** use this class directly; instead use FireSelfAttention with ``num_heads = 1`` if you need it.

        Args:
            dim_model (int): The embedding dimension of the input vectors, as above.
            kdim (int): The dimension of the query, key, and value vectors, computed as ``kdim = dim_model // num_heads``.
            hidden_size (int): The dimension of the MLP layers in each attention head used to compute the bias matrix.
        """

        def __init__(self, dim_model: int, kdim: int, hidden_size: int) -> None:
            super().__init__()
            self.kdim = kdim

            # initialize parameter matrices
            self.W_q = nn.Linear(dim_model, kdim, bias=False)
            self.W_k = nn.Linear(dim_model, kdim, bias=False)
            self.W_v = nn.Linear(dim_model, kdim, bias=False)

            # initialize learnable scalars to "reasonable" values (these are arbitary and can be adjusted later on.)
            # c is used to modify the input of the logarithm in the phi function.
            self.c = nn.Parameter(torch.tensor(1.0))
            # L is used in the adaptive thresholding mechanism to activate progressive interpolation only for long contexts.
            self.L = nn.Parameter(torch.tensor(2.0))

            # initialize learnable continuous function
            self.f_theta = nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        # concave function to amplify differences among local positions
        def phi(self, c: nn.Parameter, x: int | torch.Tensor) -> torch.Tensor:
            return torch.log1p(c * x)

        def forward(self, src: torch.Tensor) -> torch.Tensor:
            """
            Args:
                src (torch.Tensor): Input tensor with shape ``[batch_size, seq_length, dim_model]``

            Returns:
                torch.Tensor: Output tensor of shape ``[batch_size, seq_length, kdim]``
            """
            # Assuming src has shape (batch_size, seq_length, dim_model)
            batch_size, seq_length = src.shape[0:2]

            # constrain c to be > 0
            c = torch.nn.functional.softplus(self.c)

            # compute bias matrix
            # below, i is the query position and j is the key position, 0 <= i - j < i
            bias = torch.zeros(seq_length, seq_length)
            for i in range(1, seq_length):
                for j in range(0, i):
                    # we have to use i + 1 in the denominator to compensate for 0-based indexing
                    bias[i, j] = self.phi(c, i - j) / self.phi(
                        c, torch.maximum(self.L, torch.tensor(i + 1))
                    )
            # apply MLP to bias matrix
            bias = self.f_theta(bias.unsqueeze(2)).squeeze(2)
            # add causal mask
            lookahead_mask = torch.ones(seq_length, seq_length, dtype=torch.bool).triu(
                diagonal=1
            )
            bias.masked_fill_(lookahead_mask, float("-inf"))
            # repeat bias matrix for batch_size
            bias = bias.repeat(batch_size, 1, 1)

            # get Query, Key, and Value matrices for each sequence
            q = self.W_q(src)
            k = self.W_k(src)
            v = self.W_v(src)

            # calculate attention scores
            k_t = torch.transpose(k, 1, 2)
            attn_logits = torch.bmm(q, k_t) / (self.kdim**0.5)
            attn_logits = attn_logits + bias
            attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
            attn_outputs = torch.bmm(attn_weights, v)
            return attn_outputs

    # End of the inner class for a single attention head

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Input tensor with shape ``[batch_size, seq_length, dim_model]``

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_length, dim_model]`` with multi-head attention
            and FIRE relative positional encoding applied.

        Example:

            >>> import torch
            >>> from torchtune.modules import FireSelfAttention
            >>>
            >>> # instantiate module
            >>> test_layer = FireSelfAttention(dim_model=512, num_heads=8, hidden_size=32)
            >>>
            >>> # input tensor; FireSelfAttention expects a format of (batch_size, seq_len, dim_model)
            >>> x = torch.randn(64, 20, 512)
            >>>
            >>> # get output of attention layer with FIRE positional encoding
            >>> y = test_layer(x)
            >>> print(y.shape)
            torch.Size([64, 20, 512])
        """
        # src should have shape (batch_size, seq_length, dim_model)
        # Pass src through the attention heads
        attn_results = [attn_head(src) for attn_head in self.attention_heads]
        # concatenate results
        attn_results = torch.cat(attn_results, dim=-1)
        # pass through final linear layer
        return self.W_o(attn_results)
