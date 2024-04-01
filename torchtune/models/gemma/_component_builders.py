# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Literal, Optional

from torch import nn

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    KVCache,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torchtune.modules.peft import LORA_ATTN_MODULES, LoRALinear
from torchtune.models.gemma._model_utils import tie_weight

"""
Component builders for the Gemma 2B models and popular variants such as LoRA.

TorchTune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``CausalSelfAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def gemma(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
) -> TransformerDecoder:
    """
    Build the decoder associated with the gemma model. This includes:
    - Token embeddings
    - num_layers number of TransformerDecoderLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    This does NOT currently include inference-time optimizations such as
    sliding-window attention

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms Default: 1e-6
        rope_base (int): base for the rotary positional embeddings. Default: 10_000

    Returns:
        TransformerDecoder: Instantiation of gemma model.
    """
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_att = CausalSelfAttention(  # TODO: check is it the same as in the paper, SdpaAttention
        embed_dim=embed_dim,    # 2048
        num_heads=num_heads,    # 8
        num_kv_heads=num_kv_heads,  # 1
        head_dim=head_dim,  # 256
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
        pos_embeddings=rope,    # TODO: check is it the same as in the paper, GemmaRotaryEmbedding
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerDecoderLayer(
        attn=self_att,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
    # tie_weight(model)
    return model


def gemma_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Gemma model.

    Args:
        dim (int): input dimension to the MLP
        hidden_dim (int): hidden dimension of the MLP
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    activation = nn.GELU(approximate="tanh")
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)


def lora_gemma(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # gemma args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
) -> TransformerDecoder:
    """
    Return a version of Gemma (an instance of :func:`~torchtune.modules.TransformerDecoder`)
    with LoRA applied to some of the linear layers in its self-attention modules.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with
        intermediate_dim (int): intermediate dimension for MLP.
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms.
        rope_base (int): base for the rotary positional embeddings. Default: 10_000
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0

    Returns:
        TransformerDecoder: Instantiation of Gemma model with LoRA applied to
        a subset of the attention projections in each layer.
    """
    self_attn = lora_gemma_self_attention(
        lora_modules=lora_attn_modules,
        head_dim=head_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        rope_base=rope_base,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    if apply_lora_to_mlp:
        mlp = lora_gemma_mlp(
            dim=embed_dim,
            hidden_dim=intermediate_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    else:
        mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = (
        LoRALinear(
            embed_dim,
            vocab_size,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, vocab_size, bias=False)
    )
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )
    # tie_weight(model)
    return model


def lora_gemma_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # CausalSelfAttention args
    head_dim: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 10_000,
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
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        max_seq_len (int): maximum sequence length the model will be run with
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        rope_base (int): base for the rotary positional embeddings. Default: 10_000
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
    q_proj = (
        LoRALinear(
            embed_dim,
            num_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "q_proj" in lora_modules
        else nn.Linear(embed_dim, num_heads * head_dim, bias=False)
    )
    k_proj = (
        LoRALinear(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "k_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    v_proj = (
        LoRALinear(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "v_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
    )
    output_proj = (
        LoRALinear(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        if "output_proj" in lora_modules
        else nn.Linear(embed_dim, embed_dim, bias=False)
    )
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = CausalSelfAttention(    # TODO: check is it the same as in the paper, SdpaAttention
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,    # TODO: check is it the same as in the paper, GemmaRotaryEmbedding
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_gemma_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
) -> FeedForward:
    """
    Return an instance of :func:`~torchtune.modules.FeedForward` with LoRA

    Args:
        dim (int): input dimension to the MLP
        hidden_dim (int): hidden dimension of the MLP
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base (bool): whether to quantize the base of the LoRA approximation.
            Default: False

    Returns:
        FeedForward: instantiation of the MLP module with LoRA applied to
    """
    gate_proj = LoRALinear(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    down_proj = LoRALinear(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    up_proj = LoRALinear(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    activation = nn.GELU(approximate="tanh")
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)
