# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor

from torchtune.modules.attention import CausalSelfAttention
from torchtune.modules.feed_forward import FeedForward

from torchtune.modules.peft.lora import LoRALinear
from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings
from torchtune.modules.rms_norm import RMSNorm
from torchtune.modules.transformer import TransformerDecoder, TransformerDecoderLayer


def llama_feedforward(dim: int, hidden_dim: int, multiple_of: int = 256):
    """Notably, this utilizes a variant of GLU called SwiGLU, a combination of Swish
    and Gated Linear Units, formulated in https://arxiv.org/pdf/2002.05202.pdf (Shazeer 2020)."""
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * hidden_dim / 3)

    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    w1 = nn.Linear(dim, hidden_dim, bias=False)
    w2 = nn.Linear(hidden_dim, dim, bias=False)
    w3 = nn.Linear(dim, hidden_dim, bias=False)
    activation = F.silu
    return FeedForward(linear1=w1, linear2=w2, linear3=w3, activation=activation)


def llama_transformer_decoder_layer(
    embed_dim: int,
    num_heads: int,
    max_seq_len: int = 4096,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
):
    self_attention = llama_self_attention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        max_batch_size=max_batch_size,
    )
    mlp = llama_feedforward(dim=embed_dim, hidden_dim=embed_dim)
    attn_norm = RMSNorm(dim=embed_dim)
    ff_norm = RMSNorm(dim=embed_dim)
    return TransformerDecoderLayer(
        self_attention=self_attention,
        mlp=mlp,
        attn_norm=attn_norm,
        ff_norm=ff_norm,
    )


def llama_transformer_decoder(
    vocab_size: int,
    embed_dim: int,
    # transformer layer params
    num_layers: int,
    num_heads: int,
    max_seq_len: int = 4096,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    # RMS Norm params
    norm_eps: float = 1e-6,
    max_batch_size: Optional[int] = None,
):
    token_embeddings = nn.Embedding(vocab_size, embed_dim)
    layer = TransformerDecoderLayer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        max_batch_size=self.max_batch_size,
    )
    norm = RMSNorm(embed_dim, eps=norm_eps)
    output = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        token_embeddings=token_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=norm,
        output=output,
    )


# Note: we can also use dataclass to simplify passing LoRA args around
def llama_lora_self_attention(
    self,
    embed_dim: int,
    num_heads: int,
    lora_rank: int,
    lora_alpha: float,
    max_seq_len: int = 4096,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    lora_use_bias: bool = False,
    lora_dropout: float = 0.0,
):

    # Output dimension of the qkv projection matrix depends on the
    # total number of heads and the dimension of each head.
    # For MHA this is simply 3 * embed_dim since num_kv_heads = num_heads
    head_dim = embed_dim // num_heads
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim

    qkv_proj = LoRALinear(
        embed_dim, lora_rank, qkv_dim, lora_alpha, lora_use_bias, lora_dropout
    )
    output_proj = LoRALinear(
        embed_dim, lora_rank, embed_dim, lora_alpha, lora_use_bias, lora_dropout
    )

    # Build the RoPE cache
    rope_embeddings = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len)

    qkv_proj = nn.Linear(embed_dim, qkv_dim, bias=False)
    rope_embeddings = RotaryPositionalEmbeddings(embed_dim)
    sdpa = nn.functional.scaled_dot_product_attention
    return LlamaSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_proj=qkv_proj,
        out_proj=out_proj,
        rope_embeddings=rope_embeddings,
        sdpa=sdpa,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
        attn_dropout=attn_dropout,
        max_batch_size=max_batch_size,
    )


# Note that some LoRA configs do not even apply to FFN
# In that case we can just use llama_feedforward directly
def llama_lora_feedforward(
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    multiple_of: int = 256,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    lora_use_bias: bool = False,
    lora_dropout: float = 0.0,
):

    # TODO: if we go this route and have this calculation in a bunch of places
    # just make it a util
    hidden_dim = 4 * int(2 * hidden_dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    w1 = nn.LoRALinear(
        dim, lora_rank, hidden_dim, lora_alpha, bias=lora_use_bias, dropout=lora_dropout
    )
    w2 = nn.LoRALinear(
        hidden_dim, lora_rank, dim, lora_alpha, bias=lora_use_bias, dropout=lora_dropout
    )
    w3 = nn.Linear(
        dim, lora_rank, hidden_dim, lora_alpha, bias=lora_use_bias, dropout=lora_dropout
    )
    activation = F.silu
    return FeedForward(linear1=w1, linear2=w2, linear3=w3, activation=activation)


# Q: should we expose these params at all levels of builder functions?
def llama_lora_transformer_decoder_layer(
    # Should also follow Nicolas's suggestion and go the keywords-only route here
    embed_dim: int,
    num_heads: int,
    lora_rank: int,
    lora_alpha: float,
    max_seq_len: int = 4096,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    lora_use_bias: bool = False,
    lora_dropout: float = 0.0,
):
    self_attention = llama_lora_self_attention(
        embed_dim,
        num_heads,
        lora_rank,
        lora_alpha,
        max_seq_len,
        num_kv_heads,
        attn_dropout,
        max_batch_size,
        lora_use_bias,
        lora_dropout,
    )
    mlp = llama_lora_feedforward(
        embed_dim,
        embed_dim,
        lora_rank,
        lora_alpha,
        num_kv_heads,
        attn_dropout,
        max_batch_size,
        lora_use_bias,
        lora_dropout,
    )
    attn_norm = RMSNorm(dim=embed_dim)
    ff_norm = RMSNorm(dim=embed_dim)
    return TransformerDecoderLayer(
        self_attention=self_attention,
        mlp=mlp,
        attn_norm=attn_norm,
        ff_norm=ff_norm,
    )


def llama_lora_transformer_decoder(
    vocab_size,
    embed_dim,
    # transformer layer params
    num_layers,
    num_heads,
    max_seq_len,
    # RMS Norm params
    norm_eps,
    lora_rank: int,
    lora_alpha: float,
    num_kv_heads: Optional[int] = None,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    lora_use_bias: bool = False,
    lora_dropout: float = 0.0,
):
    token_embeddings = nn.Embedding(vocab_size, embed_dim)
    layer = llama_lora_transformer_decoder_layer(
        embed_dim,
        num_heads,
        lora_rank,
        lora_alpha,
        max_seq_len,
        num_kv_heads,
        attn_dropout,
        max_batch_size,
        num_kv_heads,
        attn_dropout,
        max_batch_size,
        lora_use_bias,
        lora_dropout,
    )
    norm = RMSNorm(embed_dim, eps=norm_eps)
    output = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        token_embeddings=token_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=norm,
        output=output,
    )
