# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Literal, Optional

from torch import nn

from torchtune.models.llama2 import _scale_hidden_dim_for_mlp

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    KVCache,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerDecoderLayer,
)

from torchtune.modules.peft import LoRALinear

# Modules from CausalSelfAttention that LoRA can be applied to
LORA_ATTN_MODULES = Literal["q_proj", "k_proj", "v_proj", "output_proj"]

def lora_llama2_7b(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    max_batch_size: Optional[int] = None,
) -> TransformerDecoder:
    """Builder for creating a Llama2 model initialized w/ the default 7b parameter values.
    From https://arxiv.org/abs/2307.09288, these default values are:
    - vocab_size: 32,000
    """
    return lora_llama2(
        lora_attn_modules=lora_attn_modules,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        embed_dim=4096,
        max_seq_len=4096,
        max_batch_size=max_batch_size,
        attn_dropout=0.0,
        norm_eps=1e-6, #TODO: PICK A VALUE HERE
        # Based on
        # https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L41-L42
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )


def _lora_llama_self_attention(
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


def lora_llama2(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    *,
    # llama2 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    max_batch_size: Optional[int] = None,
    norm_eps: float = 1e-6,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
) -> TransformerDecoder:
    """
    Return a version of Llama2 (an instance of :func:`~torchtune.modules.TransformerDecoder`)
    with LoRA applied to some of the linear layers in its self-attention modules.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        max_batch_size (Optional[int]): maximum batch size to be passed to :func:`~torchtune.modules.KVCache`
        norm_eps (float): epsilon in RMS norms.
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0

    Returns:
        TransformerDecoder: Instantiation of Llama2 model with LoRA applied to
        a subset of the attention projections in each layer.

    """
    self_attn = _lora_llama_self_attention(
        lora_modules=lora_attn_modules,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        max_batch_size=max_batch_size,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # TODO: MLP class is not LoRA-ready yet
    hidden_dim = _scale_hidden_dim_for_mlp(embed_dim)
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
