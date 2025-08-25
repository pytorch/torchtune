# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable
from torch import nn

from torchtune.models.qwen2_5_vision._encoder import (
    Qwen25VisionPatchEmbed,
    Qwen25VLPatchMerger,
    Qwen25VisionTransformer,
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
    Qwen25VisionRotaryPositionalEmbeddings,
)

"""
Component builders for the Qwen 2.5 VL model and its constituent models.
torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. 
"""


def qwen2_5_vl_decoder(
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
    mrope_section: list[int] = [16, 24, 24],
    tie_word_embeddings: bool = False,
) -> TransformerDecoder:
    """
    same architecture as Qwen 2.5 text decoder, just with multimodal RoPE (M-RoPE)
    for handling 3D position embeddings in vision-language sequences.
    
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
        mrope_section (list[int]): MRoPE sections [temporal, height, width]. Default: [16, 24, 24]
        tie_word_embeddings (bool): Whether to tie word embeddings. Default: False
        
    Returns:
        TransformerDecoder: Text decoder with multimodal RoPE support.
    """
    head_dim = embed_dim // num_heads 

    rope = Qwen25VLRotaryPositionalEmbeddings(
            head_dim=head_dim,
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
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            is_causal=True,
        )

        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, intermediate_dim, bias=False),
            up_proj=nn.Linear(embed_dim, intermediate_dim, bias=False),
            down_proj=nn.Linear(intermediate_dim, embed_dim, bias=False),
            activation=nn.SiLU(),
        )

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
    full_att_block_indexes: list[int],
    temporal_patch_size: int,
) -> Qwen25VisionTransformer:
    """
    Build the vision encoder for Qwen2.5-VL model, including vision-language merger.

    Args:
        embed_dim (int): Embedding dimension.
        num_layers (int): Number of transformer layers.
        activation (Callable): Activation function.
        intermediate_size (int): Intermediate size.
        num_heads (int): Number of attention heads.
        in_channels (int): Number of input channels.
        out_hidden_size (int): Output hidden size.
        patch_size (int): Patch size.
        spatial_merge_size (int): Spatial merge size.
        window_size (int): Window size.
        full_att_block_indexes (list[int]): Full attention block indexes.
        temporal_patch_size (int): Temporal patch size.

    Returns:
        Qwen25VisionTransformer: Instantiation of Qwen2.5-VL vision transformer.
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    rope = Qwen25VisionRotaryPositionalEmbeddings(head_dim // 2, spatial_merge_unit=spatial_merge_size**2)
    attn_bias = True

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
        activation=activation,
    )
    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(embed_dim, eps=1e-6),
        mlp_norm=RMSNorm(embed_dim, eps=1e-6),
        sa_scale=None,
        mlp_scale=None,
    )
    
    patch_embed = Qwen25VisionPatchEmbed(
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    merger = Qwen25VLPatchMerger(
        dim=out_hidden_size,
        context_dim=embed_dim,
        spatial_merge_size=spatial_merge_size,
    )

    return Qwen25VisionTransformer(
        patch_size=patch_size,
        num_layers=num_layers,
        layer=transformer_layer,
        patch_embed=patch_embed,
        patch_merger=merger,
        full_att_block_indexes=full_att_block_indexes,
        spatial_merge_size=spatial_merge_size,
        window_size=window_size,
    )
