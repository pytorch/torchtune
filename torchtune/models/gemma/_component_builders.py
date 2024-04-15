# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn

from torchtune.modules import (
    CausalSelfAttention,
    FeedForward,
    RotaryPositionalEmbeddings,
    TransformerDecoderLayer,
)
from torchtune.models.gemma.rms_norm import GemmaRMSNorm
from torchtune.models.gemma.transformer import GemmaTransformerDecoder

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
    norm_embeddings: bool = True,
) -> GemmaTransformerDecoder:
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
        norm_embeddings (bool): whether to apply layer norm before the self-attention
            and mlp layers. Default: True

    Returns:
        GemmaTransformerDecoder: Instantiation of gemma model.
    """
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_att = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)
    layer = TransformerDecoderLayer(
        attn=self_att,
        mlp=mlp,
        sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    model = GemmaTransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        norm_embeddings=norm_embeddings,
    )
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
