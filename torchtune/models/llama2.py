# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

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


def llama2_7b(vocab_size: int) -> TransformerDecoder:
    return TransformerDecoder(
        vocab_size=vocab_size,
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        max_seq_len=2048,
        norm_eps=1e-5,
    )


def small_test_ckpt(vocab_size: int) -> TransformerDecoder:
    return TransformerDecoder(
        vocab_size=32_000,
        num_layers=4,
        num_heads=16,
        embed_dim=256,
        max_seq_len=2048,
        norm_eps=1e-5,
        num_kv_heads=8,
    )


def llama2_tokenizer(path: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer


class Llama2FeedForward(nn.Module):
    """Notably, this utilizes a variant of GLU called SwiGLU, a combination of Swish
    and Gated Linear Units, formulated in https://arxiv.org/pdf/2002.05202.pdf (Shazeer 2020)."""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
        # parameters and computation constant
        hidden_dim = 4 * int(2 * hidden_dim / 3)

        # Round hidden dimension to nearest multiple of `multiple_of`
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.ff = FeedForward(
            dim=dim, hidden_dim=hidden_dim, linear=nn.Linear, activation=F.silu
        )

    def forward(self, x):
        return self.ff(x)


class Llama2DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__()
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
        mlp = Llama2FeedForward(dim=embed_dim, hidden_dim=embed_dim)
        attn_norm = RMSNorm(dim=embed_dim)
        ff_norm = RMSNorm(dim=embed_dim)
        self.layer = TransformerDecoderLayer(
            self_attention=self_attn,
            mlp=mlp,
            sa_norm=attn_norm,
            mlp_norm=ff_norm,
        )

    def forward(self, x, mask: Optional[Tensor] = None, curr_pos: int = 0) -> Tensor:
        return self.layer(x, mask, curr_pos)


class Llama2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        # Transformer layer params
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 4096,
        num_kv_heads: Optional[int] = None,
        attn_dropout: float = 0.0,
        # RMS Norm param
        norm_eps: float = 1e-6,
        # Optional KV cache param
        max_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.max_batch_size = max_batch_size
        token_embeddings = nn.Embedding(vocab_size, embed_dim)
        layer = Llama2DecoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            max_batch_size=max_batch_size,
        )
        norm = RMSNorm(embed_dim, eps=norm_eps)
        output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.model = TransformerDecoder(
            token_embeddings=token_embeddings,
            layer=layer,
            num_layers=num_layers,
            norm=norm,
            output=output,
        )

    def forward(self, tokens: Tensor, curr_pos: int = 0) -> Tensor:
        seq_len = tokens.size(1)
        mask = None
        if seq_len > 1 and self.max_batch_size is not None:
            mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=curr_pos + 1)
        return self.model(tokens, mask, curr_pos)
