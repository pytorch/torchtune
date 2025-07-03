# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Callable
from torch import nn

from torchtune.models.qwen2_5_vision._encoder import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLPatchMerger,
    Qwen2_5_VisionTransformer,
)
from torchtune.modules import (
    MultiHeadAttention,
    RMSNorm,
    TransformerSelfAttentionLayer,
    FeedForward,
    TransformerDecoder,
    TiedLinear,
)
from torchtune.models.qwen2_5_vision._positional_embeddings import (
    Qwen25VLRotaryPositionalEmbeddings,
    Qwen2_5_VisionRotaryEmbedding,
)

"""
Component builders for the Qwen 2.5 VL model and its constituent models.
torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``GroupedQueryAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


def qwen2_5_vl_text_decoder(
    vocab_size: int = 152064,
    num_layers: int = 28,
    num_heads: int = 28,
    num_kv_heads: int = 4,
    embed_dim: int = 3584,
    intermediate_dim: int = 18944,
    max_seq_len: int = 32768,
    attn_dropout: float = 0.0,
    rope_base: float = 1000000.0,
    norm_eps: float = 1e-6,
    mrope_section: List[int] = [16, 24, 24],
    tie_word_embeddings: bool = False,
) -> TransformerDecoder:
    """
    Build the text decoder for Qwen2.5-VL model following TorchTune patterns.
    
    This builds a standard transformer decoder with multimodal RoPE (MRoPE)
    for handling 3D position embeddings in vision-language sequences.
    
    To use with 3D position_ids, pass them as the `input_pos` parameter
    when calling the decoder forward method.
    
    Args:
        vocab_size (int): Size of vocabulary. Default: 152064
        num_layers (int): Number of transformer layers. Default: 28
        num_heads (int): Number of query heads. Default: 28
        num_kv_heads (int): Number of key/value heads (GQA). Default: 4
        embed_dim (int): Embedding dimension. Default: 3584
        intermediate_dim (int): MLP intermediate dimension. Default: 18944
        max_seq_len (int): Maximum sequence length. Default: 32768
        attn_dropout (float): Attention dropout rate. Default: 0.0
        rope_base (float): RoPE base frequency. Default: 1000000.0
        norm_eps (float): RMS norm epsilon. Default: 1e-6
        mrope_section (List[int]): MRoPE sections [temporal, height, width]. Default: [16, 24, 24]
        
    Returns:
        TransformerDecoder: Text decoder with multimodal RoPE support
        
    Example:
        >>> decoder = qwen2_5_vl_text_decoder()
        >>> # For multimodal usage, pass 3D position_ids as input_pos
        >>> output = decoder(tokens, input_pos=position_ids_3d)  # position_ids_3d: [3, b, s]
    """
    head_dim = embed_dim // num_heads

    rope = Qwen25VLRotaryPositionalEmbeddings(
            dim=head_dim,
            mrope_section=mrope_section,
            base=rope_base,
            max_seq_len=max_seq_len,
        )
    # Create layers
    layers = nn.ModuleList()
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True), 
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            is_causal=True,
        )

        mlp = qwen2_5_vl_text_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )

        layers.append(layer)
    
    # Create embeddings and output projection
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    if tie_word_embeddings:
        output_proj = TiedLinear(embed_dim, vocab_size, bias=False)
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



def qwen2_5_vision_encoder(
    embed_dim: int,
    num_layers: int,
    activation: Callable,
    intermediate_size: int,
    num_heads: int,
    in_channels: int,
    out_hidden_size: int,
    patch_size: int,
    spatial_merge_size: int,
    window_size: int,
    fullatt_block_indexes: List[int],
    temporal_patch_size: int,
) -> Qwen2_5_VisionTransformer:
    """
    TODO: docstring 
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    rope = Qwen2_5_VisionRotaryEmbedding(head_dim // 2, spatial_merge_unit=spatial_merge_size**2)
    attn_bias = True

    # transformer layer # TODO: check if need custom attn
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        pos_embeddings=rope,
        attn_dropout=0.0,
        is_causal=False,
    )
    mlp = FeedForward(
        gate_proj=nn.Linear(embed_dim, intermediate_size, bias=True),
        down_proj=nn.Linear(intermediate_size, embed_dim, bias=True),
        up_proj=nn.Linear(embed_dim, intermediate_size, bias=True),
        activation=activation(),
    )
    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=1e-6),
        mlp_norm=RMSNorm(embed_dim, eps=1e-6),
        sa_scale=None,
        mlp_scale=None,
    )
    
    patch_embed = Qwen2_5_VisionPatchEmbed(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    merger = Qwen2_5_VLPatchMerger(
        dim=out_hidden_size,
        context_dim=embed_dim,
        spatial_merge_size=spatial_merge_size,
    )

    return Qwen2_5_VisionTransformer(
        patch_size=patch_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        in_channels=in_channels,
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
        fullatt_block_indexes=fullatt_block_indexes,
        layer=transformer_layer,
        patch_embed=patch_embed,
        patch_merger=merger,
    )

def qwen2_5_vl_text_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Qwen2.5 VL model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)

def qwen2_5_vision_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    activation: Callable = nn.SiLU,
    mlp_bias: bool = True,
) -> Qwen2_5_VisionMLP:
    gate_proj = nn.Linear(in_dim, hidden_dim, bias=mlp_bias)
    down_proj = nn.Linear(hidden_dim, out_dim, bias=mlp_bias)
    up_proj = nn.Linear(hidden_dim, out_dim, bias=mlp_bias)
    return Qwen2_5_VisionMLP(
        gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation
    )