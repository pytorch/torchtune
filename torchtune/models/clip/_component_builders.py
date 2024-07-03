from typing import List, Optional, Callable

import torch
from torchtune.modules import VisionTransformer, CLSProjection
from torchtune.models.clip._position_embeddings import TokenPositionalEmbedding, TiledTokenPositionalEmbedding, TilePositionalEmbedding

import logging

logger = logging.getLogger(__name__)

def clip_vision_encoder(
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    act_layer: Callable,
    indices_return_hidden: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    tile_size: int = 512,
    patch_size: int = 14,
    max_num_tiles: int = 4,
    mlp_ratio: float = 4.0,
    in_channels: int = 3,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    cls_output_dim: int = 512,
) -> VisionTransformer:
    """
    Build the vision encoder associated with the clip model. This includes:
    - TransformerEncoderLayer
    - positional embeddings
    - cls projection (optional)

    Args:
        embed_dim (int): The dimensionality of each patch embedding (token).
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        act_layer (Callable): The activation function used in the transformer layers.
        indices_return_hidden (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, indices_return_hidden = [0, 3] will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for patch_size = 40, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        mlp_ratio (float): The ratio of the feedforward network size to the size of the transformer layers.
        in_channels (int): The number of image input channels.
        attn_dropout (float): The dropout rate applied to the attention weights.
        norm_eps (float): The epsilon used for layer normalization to prevent division by zero.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        
    Returns:
        A `VisionTransformer` object.
    """

    patch_grid_size = tile_size // patch_size

    cls_projection = CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim) if output_cls_projection else None
    
    # TODO (Felipe): Replace with torchtune native encoder module
    transformer_layer = torch.nn.TransformerEncoderLayer(
        d_model=embed_dim, 
        nhead=num_heads, 
        dim_feedforward=int(mlp_ratio * embed_dim), 
        dropout=attn_dropout, 
        activation=act_layer, 
        layer_norm_eps=norm_eps, 
        batch_first=True, 
        norm_first=True, 
        bias=True)

    # position embeddings
    if max_num_tiles == 1:
        logger.info("Found max_num_tiles=1. Setting tile_pos_embed to None and using only token_pos_embedding.")
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(
            embed_dim=embed_dim, 
            patch_size=patch_size, 
            tile_size=tile_size)
    else:
        logger.info(f"Found {max_num_tiles=}. Instantiating tile_pos_embedding and token_pos_embedding.")
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
        indices_return_hidden=indices_return_hidden,
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
    )
