# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional

from torch import nn

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp

from torchtune.modules import (
    MultiHeadAttention,
    FeedForward,
    FrozenNF4Linear,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import _register_reparametrize_state_dict_hooks

from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Llama3 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


# ------------------ Vanilla Llama3 ------------------

def llama3(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.

    Returns:
        TransformerDecoder: Instantiation of Llama3 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
    layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

def llama3_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False) if not quantize_base else FrozenNF4Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)



# ------------------ LoRA Llama3 ------------------


def lora_llama3(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # llama3 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    intermediate_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 500000.0,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Return a version of Llama3 (an instance of :func:`~torchtune.modules.TransformerDecoder`)
    with LoRA applied based on the passed in configuration.

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
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        TransformerDecoder: Instantiation of Llama3 model with LoRA applied to
        a subset of the attention projections in each layer.

    """

    self_attn = lora_llama3_self_attention(
        lora_modules=lora_attn_modules,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        rope_base=rope_base,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        quantize_base=quantize_base,
        use_dora=use_dora,
    )

    hidden_dim = intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    if apply_lora_to_mlp:
        mlp = lora_llama3_mlp(
            dim=embed_dim,
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            quantize_base=quantize_base,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
        )
    else:
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base)

    layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    # TODO: quantize_base is not applied to final output_proj currently.
    adapter_cls = DoRALinear if use_dora else LoRALinear
    output_proj = (
        adapter_cls(embed_dim, vocab_size, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        if apply_lora_to_output
        else nn.Linear(embed_dim, vocab_size, bias=False)
    )
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        _register_reparametrize_state_dict_hooks(model)

    return model


def lora_llama3_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # MultiHeadAttention args
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: float = 500000.0,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
) -> MultiHeadAttention:
    """
    Return an instance of :func:`~torchtune.modules.MultiHeadAttention` with LoRA
    applied to a subset of its linear layers

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to. Options are ``{"q_proj", "k_proj", "v_proj",
            "output_proj"}``.
        embed_dim (int): embedding dimension for self-attention
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).

    Returns:
        MultiHeadAttention: instantiation of self-attention module with LoRA
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
    adapter_cls = DoRALinear if use_dora else LoRALinear
    q_proj = (
        adapter_cls(
            embed_dim,
            num_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "q_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_heads * head_dim, bias=False)
        )
    )
    k_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "k_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        )
    )
    v_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "v_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        )
    )
    output_proj = (
        adapter_cls(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "output_proj" in lora_modules
        else (
            nn.Linear(embed_dim, embed_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, embed_dim, bias=False)
        )
    )
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_llama3_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    quantize_base: bool = False,
    use_dora: bool = False,
) -> FeedForward:
    adapter_cls = DoRALinear if use_dora else LoRALinear
    gate_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    down_proj = adapter_cls(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    up_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    return FeedForward(
        gate_proj=gate_proj,
        down_proj=down_proj,
        up_proj=up_proj,
    )
