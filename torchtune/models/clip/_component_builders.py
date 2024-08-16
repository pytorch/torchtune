from typing import List, Optional

import torch
from torch import nn

from torchtune.modules.vision_transformer import VisionTransformer, CLSProjection
from torchtune.models.clip._position_embeddings import TokenPositionalEmbedding, TiledTokenPositionalEmbedding, TilePositionalEmbedding

from torchtune.modules import (
    TransformerSelfAttentionLayer,
    MultiHeadAttention,
    TanhGate,
    FeedForward,
    Fp32LayerNorm
) 

def clip_vision_encoder(
    tile_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    cls_output_dim: int = 512,
    out_indices: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    max_num_tiles: int = 4,
    in_channels: int = 3,
) -> VisionTransformer:
    """
    Builds the vision encoder associated with the clip model. This includes:
    
    - TransformerEncoderLayer
    - positional embeddings
    - CLS projection (optional)

    For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        embed_dim (int): The dimensionality of each patch embedding (token).
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.

    Returns:
        A `VisionTransformer` object.
    """

    cls_projection = CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim) if output_cls_projection else None
    
    # transformer block
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * embed_dim)
    head_dim = embed_dim // num_heads
    num_kv_heads = num_heads

    # transformer layer
    self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=True),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=True),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=True),
            pos_embeddings=None,
            attn_dropout=0.0,
            is_causal=False,
        )

    mlp = clip_mlp(
        in_dim=embed_dim,
        hidden_dim=int(mlp_ratio * embed_dim),
        out_dim=embed_dim,
        activation=torch.nn.SiLU(),
    )

    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        attn_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
        mlp_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
        attn_scale=None,
        mlp_scale=None,
    )

    # position embeddings
    if max_num_tiles == 1:
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(
            embed_dim=embed_dim, 
            patch_size=patch_size, 
            tile_size=tile_size)
    else:
        pre_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=max_num_tiles, embed_dim=embed_dim)
        post_tile_pos_embed = TilePositionalEmbedding(max_num_tiles=max_num_tiles, embed_dim=embed_dim)
        token_pos_embedding = TiledTokenPositionalEmbedding(
            max_num_tiles=max_num_tiles, 
            embed_dim=embed_dim, 
            patch_size=patch_size, 
            tile_size=tile_size)

    return VisionTransformer(
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        pre_tile_pos_embed=pre_tile_pos_embed,
        post_tile_pos_embed=post_tile_pos_embed,
        cls_projection=cls_projection,
        out_indices=out_indices,
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
    )


def clip_mlp(in_dim: int, out_dim: int, hidden_dim: int, activation: nn.Module, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the clip model.
    """
    gate_proj = nn.Linear(in_dim, hidden_dim) if not quantize_base else FrozenNF4Linear(in_dim, hidden_dim)
    down_proj = nn.Linear(hidden_dim, out_dim, bias=False) if not quantize_base else FrozenNF4Linear(hidden_dim, out_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=None, activation=activation)
