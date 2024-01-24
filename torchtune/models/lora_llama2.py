# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Optional

from torch import nn

from torchtune.modules import CausalSelfAttention, KVCache, RotaryPositionalEmbeddings

from torchtune.modules.peft import LoRALinear

# Modules from CausalSelfAttention that LoRA can be applied to
LORA_ATTN_MODULES = Literal["q_proj", "k_proj", "v_proj", "output_proj"]


def lora_llama_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # CausalSelfAttention args
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
) -> CausalSelfAttention:
    """
    Return an instance of :func:`~torchtune.modules.CausalSelfAttention` with LoRA
    applied to a subset of its linear layers

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to. Options are ``{"q_proj", "k_proj", "v_proj",
            "output_proj"}``.
        embed_dim (int): embedding dimension for self-attention
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        max_batch_size (Optional[int]): maximum batch size to be passed to :func:`~torchtune.modules.KVCache`
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0

    Returns:
        CausalSelfAttention: instantiation of self-attention module with LoRA
        applied to a subset of Q, K, V, output projections.

    Raises:
        ValueError: If lora_modules arg is an empty list
    """
    if not lora_modules:
        raise ValueError(
            f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules"
        )

    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    kv_cache = (
        KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=num_heads,
            head_dim=head_dim,
        )
        if max_batch_size is not None
        else None
    )
    q_proj = (
        LoRALinear(embed_dim, num_heads * head_dim, rank=lora_rank, alpha=lora_alpha)
        if "q_proj" in lora_modules
        else nn.Linear(embed_dim, num_heads * head_dim, bias=False)
    )
    k_proj = (
        LoRALinear(embed_dim, num_kv_heads * head_dim, rank=lora_rank, alpha=lora_alpha)
        if "k_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    v_proj = (
        LoRALinear(embed_dim, num_kv_heads * head_dim, rank=lora_rank, alpha=lora_alpha)
        if "v_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    output_proj = (
        LoRALinear(embed_dim, embed_dim, rank=lora_rank, alpha=lora_alpha)
        if "output_proj" in lora_modules
        else nn.Linear(embed_dim, embed_dim, bias=False)
    )
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,
        kv_cache=kv_cache,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn
