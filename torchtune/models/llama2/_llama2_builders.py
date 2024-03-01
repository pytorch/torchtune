# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

from torch import nn

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    KVCache,
    RMSNorm,
    RotaryPositionalEmbeddings,
    Tokenizer,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torchtune.models.llama2._model_utils import scale_hidden_dim_for_mlp


def llama2_7b(max_batch_size: Optional[int] = None) -> TransformerDecoder:
    """Builder for creating a Llama2 model initialized w/ the default 7b parameter values.
    From https://arxiv.org/abs/2307.09288, these default values are:
    - vocab_size: 32,000
    - embed_dim: 4,096
    - num_layers: 32
    - num_heads: 32
    - num_kv_heads: 32
    - max_seq_len: 4,096
    - norm_eps: 1e-5

    Args:
        max_batch_size (Optional[int]): Maximum batch size to be passed to KVCache.

    Returns:
        A ``TransformerDecoder`` instance of the Llama2 model.
    """
    return llama2(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        max_batch_size=max_batch_size,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def llama2_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer


def llama2(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    norm_eps: float = 1e-5,
):
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
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=kv_cache,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    hidden_dim = scale_hidden_dim_for_mlp(embed_dim)
    mlp = FeedForward(dim=embed_dim, hidden_dim=hidden_dim, linear_class=nn.Linear)
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
