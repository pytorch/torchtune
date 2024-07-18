# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import nn, Tensor
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from torchtune.modules.kv_cache import KVCache


class CausalSelfAttention(nn.Module):
    """Multi-headed grouped query self-attention (GQA) layer introduced
    in https://arxiv.org/abs/2305.13245v1.

    GQA is a version of multiheaded attention (MHA) which uses fewer
    key/value heads than query heads by grouping n query heads for each
    key and value head. Multi-Query Attention is an extreme
    version where we have a single key and value head shared by all
    query heads.

    Following is an example of MHA, GQA and MQA with num_heads = 4

    (credit for the documentation:
    https://github.com/Lightning-AI/lit-gpt/blob/main/lit_gpt/config.py).


    ::

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
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        head_dim (int): dimension of each head, calculated by ``embed_dim`` // ``num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (nn.Module): positional embeddings layer, e.g. RotaryPositionalEmbeddings.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value.
            If not specified, then no caching is used.
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        attn_dropout (float): dropout value passed onto the
            scaled_dot_product_attention function. This argument is ignored if the
            self.training is False. Default value is 0.0.

    Raises:
        ValueError: If `num_heads` % `num_kv_heads` != 0
        ValueError: If `embed_dim` % `num_heads` != 0
        ValueError: If `attn_dropout` < 0 or > 1
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: nn.Module,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
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

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.pos_embeddings = pos_embeddings

        self.flex_attention = torch.compile(flex_attention, dynamic=False)
        # self.flex_attention = flex_attention

    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        # input has shape [b, s, d]
        bsz, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # q has shape [b, s, num_heads * head_dim]
        # k has shape [b, s, num_kv_heads * head_dim]
        # v has shape [b, s, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads

        # q: [b, s, n_kv, q_per_kv, h_d]
        # k: [b, s, n_kv, 1, h_d]
        # v: [b, s, n_kv, 1, h_d]
        q = q.view(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)

        # if needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(q, input_pos=input_pos)
        k = self.pos_embeddings(k, input_pos=input_pos)

        # [b, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # Perform flex attention calculation
        output = self.flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output)
