# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional
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

from torchtune.modules.moe import (
    GroupedExperts,
    LoRAGroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)


from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Qwen3 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def qwen3_moe(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    moe_intermediate_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    max_seq_len: int,
    head_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: float = 1_000_000.0,
    tie_word_embeddings: bool = False,
    q_proj_bias: bool = True,
    k_proj_bias: bool = True,
    v_proj_bias: bool = True,
    q_norm: bool = False,
    k_norm: bool = False,
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
        head_dim (Optional[int]): Dimension of each attention head. If not
            specified, it defaults to `embed_dim // num_heads`. In GQA, `head_dim` is not necessarily equal to
            `embed_dim // num_heads`, so this parameter allows the caller to explicitly specify a custom value.
        norm_eps (float): epsilon in RMS norms.
        rope_base (float): the base period of the RoPE embeddings.
        tie_word_embeddings (bool): whether the model's input and output word embeddings should be tied.
        q_proj_bias (bool): whether to use bias in the query projection.
        k_proj_bias (bool): whether to use bias in the key projection.
        v_proj_bias (bool): whether to use bias in the value projection.
        q_norm (bool): whether to use normalization in the query projection.
        k_norm (bool): whether to use normalization in the key projection.

    Returns:
        TransformerDecoder: Instantiation of Qwen2 model.
    """
    head_dim = head_dim or embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = Qwen2RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=q_proj_bias),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=k_proj_bias),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=v_proj_bias),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            q_norm=RMSNorm(dim=head_dim, eps=norm_eps) if q_norm else None, # norm on head_dim
            k_norm=RMSNorm(dim=head_dim, eps=norm_eps) if k_norm else None,
            kv_cache=None,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        mlp = qwen3_moe_mlp(
            dim=embed_dim,
            hidden_dim=moe_intermediate_size,
            num_experts=num_experts,
            experts_per_token=num_experts_per_tok
        )
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)

    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    if tie_word_embeddings:
        output_proj = TiedLinear(tok_embeddings)
    else:
        output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def qwen3_moe_mlp(
    dim: int,
    hidden_dim: int,
    num_experts: int = 8,
    experts_per_token: int = 1,
) -> MoE:
    """
    Build the MoE layer associated with the Qwen 3 model.

    Args:
        dim (int): Input dimension of experts.
        hidden_dim (int): Hidden dimension of experts.
        num_experts (int): Number of experts in each MoE layer. Default: 8
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.

    Returns:
        MoE: Instantiation of MoE layer.
    """
    router = TokenChoiceTopKRouter(
        gate=nn.Linear(dim, num_experts, bias=False),
        dim=dim,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        norm_topk_prob = True,
        softmax=True,
    )
    experts = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
    return MoE(
        experts=experts,
        router=router,
        scale_after_fwd=True,
    )
