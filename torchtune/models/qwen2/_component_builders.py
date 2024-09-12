# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List
from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torch import nn
from torchtune.modules.transformer import TransformerDecoder
from torchtune.models.qwen2._positional_embeddings import Qwen2RotaryPositionalEmbeddings

from torchtune.modules import (
    MultiHeadAttention,
    FeedForward,
    RMSNorm,
    TransformerSelfAttentionLayer,
    TiedLinear
)


from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Qwen2 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def qwen2(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Qwen2 model. This includes:
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
        rope_base (float): the base period of the RoPE embeddings.
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.

    Returns:
        TransformerDecoder: Instantiation of Qwen2 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    mlp = qwen2_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    if tie_word_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
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


def qwen2_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Qwen2 model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


def lora_qwen2(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # qwen2 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Return a version of Qwen2 (an instance of :func:`~torchtune.models.qwen2.transformer.Qwen2TransformerDecoder`)
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
        rope_base (float): the base period of the RoPE embeddings.
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        TransformerDecoder: Instantiation of Qwen2 model with LoRA applied to
        a subset of the attention projections in each layer.

    Raises:
        ValueError: if ``apply_lora_to_output`` and ``tie_word_embeddings``.

    """

    self_attn = lora_qwen2_self_attention(
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
        use_dora=use_dora,
        quantize_base=quantize_base,
    )

    if apply_lora_to_mlp:
        mlp = lora_qwen2_mlp(
            dim=embed_dim,
            hidden_dim=intermediate_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            quantize_base=quantize_base,
            use_dora=use_dora,
            lora_dropout=lora_dropout,
        )
    else:
        mlp = qwen2_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

    layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)

    if tie_word_embeddings:
        if apply_lora_to_output:
            raise ValueError(
                "apply_lora_to_output is incompatible with tie_word_embeddings,"
                " as there would be no output to apply lora to!"
            )
        output_proj = TiedLinear(tok_embeddings)
    else:
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
        # For QLoRA, we reparametrize 4-bit tensors to higher precision, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(
                reparametrize_as_dtype_state_dict_post_hook,
                # TODO this is clowny, figure out a better way to get what precision the rest
                # of the model is in
                dtype=tok_embeddings.weight.dtype,
                offload_to_cpu=True,
            )
        )

    return model


def lora_qwen2_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # MultiHeadAttention args
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: float = 1_000_000.0,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
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
        rope_base (float): the base period of the RoPE embeddings. Default: 1_000_000.0
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.

    Returns:
        MultiHeadAttention: instantiation of self-attention module with LoRA
        applied to a subset of Q, K, V, output projections.

    Raises:
        ValueError: If lora_modules arg is an empty list
    """
    if not lora_modules:
        raise ValueError(f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules")

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
            use_bias=True,
            quantize_base=quantize_base,
        )
        if "q_proj" in lora_modules
        else nn.Linear(embed_dim, num_heads * head_dim, bias=True)
    )
    k_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_bias=True,
            quantize_base=quantize_base,
        )
        if "k_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True)
    )
    v_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_bias=True,
            quantize_base=quantize_base,
        )
        if "v_proj" in lora_modules
        else nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True)
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
        else nn.Linear(embed_dim, embed_dim, bias=False)
    )
    rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
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
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_qwen2_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
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
