# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchtune.modules import MultiHeadAttention
from torchtune.modules.transformer import _get_clones


class T5Encoder(nn.Module):
    """
    The T5 encoder module.

    T5 paper: https://arxiv.org/abs/1910.10683

    Args:
        token_embedding (nn.Embedding): PyTorch embedding layer to place tokens in an embedding space.
        layers (Union[nn.Module, List[nn.Module], nn.ModuleList]): A single encoder layer.
        final_norm (nn.Module): Module that applies normalization to the output of the encoder
        num_heads (int): The number of attention heads.
        rel_pos_num_buckets (int): Number of discrete buckets to divide the relative positions into.
            See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`
        rel_pos_max_dist (int): Maximum distance for relative positions.
            Distances beyond this are grouped into the last bucket.
            See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`
        max_seq_len (int): The maximum sequence length (context length) of the model.
        num_layers (Optional[int]): Number of encoder layers, only define when layers is not a list.

    Raises:
        AssertionError:
            If ``num_layers`` is set and layer is a list, **or**
            ``num_layers`` is not set and layer is an ``nn.Module``.

    """

    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        layers: Union[nn.Module, List[nn.Module], nn.ModuleList],
        final_norm: nn.Module,
        num_heads: int,
        rel_pos_num_buckets: int,
        rel_pos_max_dist: int,
        max_seq_len: int,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.final_norm = final_norm
        self.max_seq_len = max_seq_len
        self.relative_position_bias = T5EncoderRelativePositionBias(
            num_buckets=rel_pos_num_buckets,
            max_dist=rel_pos_max_dist,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )

        self.layers = None
        if isinstance(layers, nn.ModuleList):
            self.layers = layers
        elif isinstance(layers, list):
            self.layers = nn.ModuleList(layers)
        else:
            if not isinstance(layers, nn.Module):
                raise AssertionError("num_layers is defined, layers must be a module")
            if num_layers is None:
                raise AssertionError("num_layers is not defined, layers must be a list")
            self.layers = _get_clones(layers, num_layers)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape ``[bsz, max_seq_len]``

        Returns:
            Tensor: output tensor with shape [bsz, max_seq_len, embed_dim]

        Raises:
            ValueError: if seq_len of tokens is bigger than max_seq_len
        """
        # Input validation
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # Input embedding [bsz, max_seq_len] -> [bsz, max_seq_len, embed_dim]
        x = self.token_embedding(tokens)

        # Bias added to the attention scores of every layer (to add relative position information)
        rel_pos_bias = self.relative_position_bias()

        # Encoder
        for layer in self.layers:
            x = layer(x, rel_pos_bias)

        return self.final_norm(x)


class T5EncoderLayer(nn.Module):
    """
    Single layer of the T5 encoder (standard transformer layer with relative position bias).

    Args:
        attn (MultiHeadAttention): Attention module.
        mlp (nn.Module): Feed-forward module.
        sa_norm (nn.Module): Normalization to be applied before self-attention.
        mlp_norm (nn.Module): Normalization to be applied before the feed-forward layer.
    """

    def __init__(
        self,
        attn: MultiHeadAttention,
        mlp: nn.Module,
        sa_norm: nn.Module,
        mlp_norm: nn.Module,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.mlp = mlp
        self.sa_norm = sa_norm
        self.mlp_norm = mlp_norm

    def forward(self, x: Tensor, rel_pos_bias: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape [bsz, seq_len, embed_dim]
            rel_pos_bias (Tensor): relative position bias with shape [1, num_heads, max_seq_len, max_seq_len]
                See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`

        Returns:
            Tensor: output tensor with shape [bsz, seq_len, embed_dim]
        """
        x = x + self.attn(self.sa_norm(x), rel_pos_bias)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class T5EncoderSelfAttention(nn.Module):
    """
    Self-attention for the T5 encoder.

    Standard self-attention with two differences:
        - No scaling factor
        - Add "relative position bias" to the attention scores.
            (See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`)

    Args:
        embed_dim (int): The model dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of the attention heads (should equal `embed_dim // num_heads`)
        q_proj (nn.Module): Projection layer for query.
        k_proj (nn.Module): Projection layer for key.
        v_proj (nn.Module): Projection layer for value.
        output_proj (nn.Module): Projection layer for output.

    Raises:
        ValueError:
            If ``embed_dim % num_heads != 0``, **or**
            if ``embed_dim // num_heads != head_dim``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if embed_dim // num_heads != head_dim:
            raise ValueError(
                f"head_dim ({head_dim}) must be equal to embed_dim // num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj

    def forward(self, x: Tensor, rel_pos_bias: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape [bsz, seq_len, embed_dim]
            rel_pos_bias (Tensor): relative position bias with shape [1, num_heads, max_seq_len, max_seq_len]
                See: :class:`~torchtune.models.t5._encoder.T5EncoderRelativePositionBias`

        Returns:
            Tensor: output tensor with shape [bsz, seq_len, embed_dim]
        """
        bsz, seq_len, embed_dim = x.shape

        # QKV projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [bsz, seq_len, embed_dim] -> [bsz, num_heads, seq_len, head_dim]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # attention with relative position bias
        attn_score = torch.matmul(q, k.transpose(-2, -1))
        attn_score += rel_pos_bias
        attn_weight = F.softmax(attn_score.float(), dim=-1).to(attn_score.dtype)
        attn_out = torch.matmul(attn_weight, v)

        # [bsz, num_heads, seq_len, head_dim] -> [bsz, seq_len, embed_dim]
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, embed_dim)

        return self.output_proj(attn_out)


class T5EncoderRelativePositionBias(nn.Module):
    """
    Computes binned birectional relative position bias for the T5 encoder.

    It places relative positions into buckets and for each bucket, learns bias values for each attention head.

    Args:
        num_buckets (int): Number of discrete buckets to divide the relative positions into.
        max_dist (int): Maximum distance for relative positions (distances beyond this are grouped into the last bucket)
        num_heads (int): Number of attention heads in the transformer.
        max_seq_len (int): Maximum sequence length (context length).
    """

    def __init__(
        self, num_buckets: int, max_dist: int, num_heads: int, max_seq_len: int
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # learnable mapping from bucket indices to bias values for each attention head
        self.embedding = nn.Embedding(num_buckets, num_heads)

        # fixed mapping from relative positions to bucket indices
        self.register_buffer(
            "relative_position_to_bucket",
            _calc_birectional_rel_pos_to_bucket(num_buckets, max_dist, max_seq_len),
            persistent=False,
        )

    def forward(self) -> Tensor:
        """
        Returns:
            torch.Tensor: relative position bias tensor with shape [1, num_heads, max_seq_len, max_seq_len]
        """
        # convert bucket numbers to bias values for each attention head
        x = self.embedding(self.relative_position_to_bucket)

        # shape [max_seq_len, max_seq_len, num_heads] -> [1, num_heads, max_seq_len, max_seq_len]
        return x.permute([2, 0, 1]).unsqueeze(0)


def _calc_birectional_rel_pos_to_bucket(
    num_buckets: int, max_dist: int, max_seq_len: int
) -> Tensor:
    """
    Calculate the mapping from relative positions to bucket indices.

    NOTE: This is for the T5 encoder (birectional), not the decoder (unidirectional).

    Args:
        num_buckets (int): Number of discrete buckets to divide the relative positions into.
        max_dist (int): Maximum distance for relative positions (distances beyond this are grouped into the last bucket)
        max_seq_len (int): Maximum sequence length (context length).

    Returns:
        Tensor: shape=[max_seq_len, max_seq_len], range=[0, num_buckets]
    """
    query_positions = torch.arange(max_seq_len, dtype=torch.long)[:, None]
    key_positions = torch.arange(max_seq_len, dtype=torch.long)[None, :]
    relative_positions = key_positions - query_positions
    abs_relative_positions = torch.abs(relative_positions)
    # relative positions shape: [max_seq_len, max_seq_len]

    # divide the buckets into half for past/present (rel pos <= 0) and half for future (rel pos > 0)
    # half of the buckets in each half are for exact relative positions
    half_num_buckets = num_buckets // 2
    max_exact = half_num_buckets // 2
    is_exact = abs_relative_positions < max_exact

    # the rest are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_not_exact = max_exact + (
        torch.log(abs_relative_positions.float() / max_exact)
        / math.log(max_dist / max_exact)
        * (half_num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_not_exact = torch.min(
        relative_position_if_not_exact,
        torch.full_like(relative_position_if_not_exact, half_num_buckets - 1),
    )

    # calculate the mapping from relative postion to bucket
    relative_position_to_bucket = (relative_positions > 0).to(
        torch.long
    ) * half_num_buckets + torch.where(
        is_exact, abs_relative_positions, relative_position_if_not_exact
    )

    return relative_position_to_bucket
