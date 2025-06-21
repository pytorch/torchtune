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
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionTransformer,
)
from torchtune.modules import (
    MultiHeadAttention,
    RMSNorm,
    TransformerSelfAttentionLayer,
    FeedForward,
    TransformerDecoder,
)
from torchtune.models.qwen2_5_vision._positional_embeddings import (
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2_5_VLCompatibleRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
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



def qwen2_5_vl_text_attention_with_standard_mha(
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float = 1000000.0,
    mrope_section: List[int] = [16, 24, 24],
    max_seq_len: int = 128000,
    attn_dropout: float = 0.0,
) -> MultiHeadAttention:
    """
    Alternative builder using standard MultiHeadAttention with compatible MRoPE.
    
    This demonstrates that we can reuse the standard MultiHeadAttention by creating
    a compatible positional embedding module that handles the multimodal RoPE logic.
    
    Args:
        embed_dim (int): Embedding dimension 
        num_heads (int): Number of query heads
        num_kv_heads (int): Number of key/value heads (for GQA)
        head_dim (int): Dimension per head
        rope_theta (float): Base for RoPE frequency computation
        mrope_section (List[int]): Multimodal RoPE sections [temporal, height, width]
        max_seq_len (int): Maximum sequence length
        attn_dropout (float): Attention dropout rate
        
    Returns:
        MultiHeadAttention: Standard attention module with compatible multimodal RoPE
        
    Note:
        When using this attention module, you must call 
        `attention.pos_embeddings.update_position_embeddings(x, position_ids)` 
        before the forward pass to set the 3D position embeddings.
    """
    rope = Qwen2_5_VLCompatibleRotaryEmbedding(
        dim=head_dim,
        mrope_section=mrope_section,
        base=rope_theta,
    )

    # TODO: figure out where/how to pass in position_ids for MRoPE

    # In hf-transfomers, in Qwen2_5_VLModel, position_ids = get_rope_index(input_ids, ...)
    # position_ids are passed into the decoder (Qwen2VLTextModel), and position_embeddings are computed from position_ids (using Qwen2VLRotaryEmbedding)
    # and each of the decoder's layers (Qwen2VLTextModel) are called with the position_embeddings
    # each decoder layer's attention module (Qwen2VLAttention) is called with the position_embeddings (as well as position_ids, but not used)
        # `cos, sin = position_embeddings`
        # `query_states, key_states` = apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, ...)`
    # each decoder layer receives the same position_embeddings
    
    return MultiHeadAttention(
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
    
    # Create layers
    layers = nn.ModuleList()
    for _ in range(num_layers):
        # Create attention with multimodal RoPE
        self_attn = qwen2_5_vl_text_attention_with_standard_mha(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_base,
            mrope_section=mrope_section,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        
        # Create MLP (following Qwen2 pattern)
        mlp = FeedForward(
            gate_proj=nn.Linear(embed_dim, intermediate_dim, bias=False),
            down_proj=nn.Linear(intermediate_dim, embed_dim, bias=False),
            up_proj=nn.Linear(embed_dim, intermediate_dim, bias=False),
        )
        
        # Create transformer layer
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    
    # Create embeddings and output projection
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
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
    {
    "depth": 32,
    "hidden_act": "silu",
    "hidden_size": 1280,
    "intermediate_size": 3420,
    "num_heads": 16,
    "in_chans": 3,
    "out_hidden_size": 3584,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "spatial_patch_size": 14,
    "window_size": 112,
    "fullatt_block_indexes": [
      7,
      15,
      23,
      31
    ],
    "tokens_per_second": 2,
    "temporal_patch_size": 2
  },
    TODO: docstring
    Raises:
        AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    # TODO: change
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
    mlp = qwen2_5_vision_mlp(
        in_dim=embed_dim,
        hidden_dim=intermediate_size,
        out_dim=embed_dim,
        activation=activation(),
        mlp_bias=True,
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