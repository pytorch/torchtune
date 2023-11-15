# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from llm.llama2.position_embeddings import RotaryPositionalEmbeddings


class KVCache(nn.Module):
    """Key-Value Cache for Self-Attention.

    Args:
        max_bsz (int): Maximum batch size supported by the model.
        max_seq_len (int): Maximum sequence length supported by the model.
        num_heads (int): Number of heads used in the model.
        head_dim (int): Dimension of each head.
    """

    def __init__(
        self,
        max_bsz: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        cache_shape = (max_bsz, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.nn.Parameter(torch.zeros(cache_shape))
        self.v_cache = torch.nn.Parameter(torch.zeros(cache_shape))

    def update(
        self, bsz: int, curr_pos: int, seq_len: int, k_val: Tensor, v_val: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Update the cache with new values and return the updated keys and values.

        Args:
            bsz (int): Batch size.
            curr_pos (int): Current position.
            seq_len (int): Sequence length.
            k_val (Tensor): Key values to be cached.
            v_val (Tensor): Value values to be cached.

        Returns:
            Tuple[Tensor, Tensor]: Updated key and value caches.
        """
        total_len = curr_pos + seq_len
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:bsz, :, curr_pos:total_len] = k_val
        v_out[:bsz, :, curr_pos:total_len] = v_val

        return k_out[:bsz, :, :total_len], v_out[:bsz, :, :total_len]


class LlamaSelfAttention(nn.Module):
    """
    Multi-headed grouped query self-attention (GQA) layer introduced
    in https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    https://github.com/Lightning-AI/lit-gpt/blob/main/lit_gpt/config.py)


    ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
      │    │    │    │         │        │                 │
    ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
      │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
            MHA                    GQA                   MQA
       n_kv_heads =4          n_kv_heads=2           n_kv_heads=1

    Args:

        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (Optional[int]): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0.
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        max_bsz_for_kv_cache (Optional[int]): Maximum batch size for kv cache. Defaults to None.

    Raises:
        ValueError: If `num_heads` % `num_kv_heads` != 0
        ValueError: If `embed_dim` % `num_heads` != 0
        ValueError: If `attn_dropout` < 0 or > 1
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        max_seq_len: int = 4096,
        max_bsz_for_kv_cache: Optional[int] = None,
    ) -> None:
        super().__init__()

        if num_kv_heads and num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads

        # Output dimension of the qkv projection matrix depends on the
        # total number of heads and the dimension of each head.
        # For MHA this is simply 3 * embed_dim since num_kv_heads = num_heads
        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

        self.qkv_proj = nn.Linear(self.embed_dim, qkv_dim, bias=False)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Build the RoPE cache
        self.max_seq_len = max_seq_len
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=self.max_seq_len
        )

        # Initialize the KV cache
        self.max_bsz_for_kv_cache = max_bsz_for_kv_cache
        if self.max_bsz_for_kv_cache is not None:
            self.kv_cache = KVCache(
                self.max_bsz_for_kv_cache,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            self.kv_cache = None

    def forward(self, x: Tensor, curr_pos: int = 0) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            curr_pos (int): current position of the token in the sequence.

        Returns:
            Tensor: output tensor with attention applied

        Raises:
            ValueError: if seq_len of x is bigger than max_seq_len

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
            - qkv_d: qkv_dim compured as (n_h + 2 * n_kv) * h_d

        TODO: A few TODOs
            - Return the attention weights
            - Control whether we apply RoPE or not with a flag
        """

        # input has shape [b, s, d]
        bsz, seq_len, _ = x.shape

        if self.max_bsz_for_kv_cache is not None and bsz > self.max_bsz_for_kv_cache:
            raise ValueError(
                f"bsz of size ({bsz}) cannot exceed max_bsz_for_kv_cache ({self.max_bsz_for_kv_cache})"
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # qkv has shape [b, s, qkv_d]
        qkv = self.qkv_proj(x)

        # create the q,k and v tensors by splitting qkv
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([self.embed_dim, kv_size, kv_size], dim=-1)

        # Reshape the tensors before we apply RoPE
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q = self.rope(q, curr_pos)
        k = self.rope(k, curr_pos)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))  # (b, n_h, s, h_d)

        # Update the KV cache if it exist
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(bsz, curr_pos, seq_len, k, v)

        # If needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            q_per_kv = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats=q_per_kv, dim=1)
            v = v.repeat_interleave(repeats=q_per_kv, dim=1)

        # Use flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output)
