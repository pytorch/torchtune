# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def llama2_7b() -> TransformerDecoder:
    """Builder functions for creating a Llama2 model initialized w/ the default 7b parameter values.

    Returns:
        A ``TransformerDecoder`` instance for the Llama2 model.
    """
    return llama2(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=2048,
        max_batch_size=32,  # Need to figure out the actual default used by Llama2
        attn_dropout=0.0,
        norm_eps=1e-5,
    )


def llama2_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer


def llama2(
    vocab_size,
    num_layers,
    num_heads,
    num_kv_heads,
    embed_dim,
    max_seq_len,
    attn_dropout=0.0,
    max_batch_size=None,
    norm_eps=1e-5,
):
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
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
        qkv_proj=nn.Linear(embed_dim, qkv_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=kv_cache,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * embed_dim / 3)
    # Round hidden dimension to nearest multiple of `multiple_of`
    multiple_of = 256
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    mlp = FeedForward(dim=embed_dim, hidden_dim=hidden_dim, linear_class=nn.Linear)
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim),
        mlp_norm=RMSNorm(dim=embed_dim),
    )
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
