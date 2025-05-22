# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional, Callable
from torchtune.modules.feedforward import FeedForward

from torch import nn

from torchtune.models.qwen2_5_vision._encoder import (
    Qwen2_5VisionEncoder,
    Qwen2_5VisionProjectionHead,
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionTransformer,
)
from torchtune.modules import (
    Fp32LayerNorm,
    MultiHeadAttention,
    RMSNorm,
    TanhGate,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook
from torchtune.modules.model_fusion import FusionEmbedding, FusionLayer


"""
Component builders for the Llama 3.2 Vision model and its constituent models.
torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``GroupedQueryAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


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
    tile_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    activation: Callable = nn.SiLU,
    cls_output_dim: int = 512,
    attn_bias: bool = True,
    use_rope: bool = False,
    out_indices: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    max_num_tiles: int = 4,
    in_channels: int = 3,
    append_cls_token: bool = False,
    use_tile_pos_embed: bool = True,
) -> Qwen2_5VisionEncoder:
    """
    Builds the vision encoder associated with the clip model. This includes:

    - TransformerEncoderLayer
    - positional embeddings
    - CLS projection (optional)

    For details, please check the documentation of
    :class:`torchtune.modules.vision_transformer.VisionTransformer`.

    Args:
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        embed_dim (int): The dimensionality of each patch embedding (token).
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        activation (Callable): The activation function to use in the MLP layer.
        cls_output_dim (int): The dimensionality of the output tensor from the CLS projection module.
        attn_bias (bool): Boolean for if to use bias in the attention module. Default True.
        use_rope (bool): If True, include 2D rope in attention in each transformer layer. Default: False
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        append_cls_token (bool): If True, adds CLS token embedding to the end of the sequence in the vision transformer.
            Default is False, which adds CLS token to the beginning of the sequence.
        use_tile_pos_embed (bool): If True, use pre-tile, post-tile, and tiled token positional embeddings, if max_num_tiles > 1.
            If False, only use standard token positional embeddings.

    Returns:
        A `VisionTransformer` object.

    Raises:
        AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
    """
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}"
        )

    head_dim = embed_dim // num_heads

    # TODO: change
    rope = (
        VisionRotaryPositionalEmbeddings(
            patch_size=patch_size,
            tile_size=tile_size,
            dim=head_dim,
            base=10_000,
            append_cls_token=append_cls_token,
        )
        if use_rope
        else None
    )

    # transformer layer
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
    mlp = qwen2_5_vision_mlp( #TODO: check params
        in_dim=embed_dim,
        hidden_dim=4 * embed_dim,
        out_dim=embed_dim,
        activation=activation(),
        mlp_bias=True,
    )
    transformer_layer = TransformerSelfAttentionLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
        mlp_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
        sa_scale=None,
        mlp_scale=None,
    )

    # position embeddings
    if use_tile_pos_embed and max_num_tiles > 1:
        pre_tile_pos_embed = TilePositionalEmbedding(
            max_num_tiles=max_num_tiles, embed_dim=embed_dim
        )
        post_tile_pos_embed = TilePositionalEmbedding(
            max_num_tiles=max_num_tiles, embed_dim=embed_dim
        )
        token_pos_embedding = TiledTokenPositionalEmbedding(
            max_num_tiles=max_num_tiles,
            embed_dim=embed_dim,
            patch_size=patch_size,
            tile_size=tile_size,
        )
    else:
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(
            embed_dim=embed_dim, patch_size=patch_size, tile_size=tile_size
        )

    return Qwen2_5_VisionTransformer(
        num_layers=num_layers,
        layer=transformer_layer,
        token_pos_embedding=token_pos_embedding,
        pre_tile_pos_embed=pre_tile_pos_embed,
        post_tile_pos_embed=post_tile_pos_embed,
        out_indices=out_indices,
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        in_channels=in_channels,
        append_cls_token=append_cls_token,
    )

def qwen2_5_vision_projection_head(
   *,
    decoder_embed_dim: int,
    clip_embed_dim: int,
    projection_embed_dim: int,
) -> Qwen2_5VisionProjectionHead:
    """
    Build the Qwen 2.5 Vision Projection Head that maps the output of the CLIP encoder
    to embeddings that can be fed into the decoder.

    Args:
        decoder_embed_dim (int): embedding dimension for the decoder.
        clip_embed_dim (int): embedding dimension for the CLIP encoder.
        projection_embed_dim (int): embedding dimension for the inner linear layers in the projection head.

    Returns:
        Qwen2_5VisionProjectionHead: Instantiation of Qwen 2.5 vision projection head.
    """
    output = nn.Sequential(
        # TODO: add layernorm
        nn.Linear(projection_embed_dim, projection_embed_dim, bias=False),
        nn.GELU(),
        nn.Linear(projection_embed_dim, decoder_embed_dim, bias=False),
    )

    return Qwen2_5VisionProjectionHead(
        output=output,
    )



def qwen2_5_vision_encoder(
    # clip encoder parameters
    *,
    patch_size: int,
    num_heads: int,
    clip_embed_dim: int,
    clip_num_layers: int,
    clip_hidden_states: Optional[List[int]],
    # projection parameters
    num_layers_projection: int,
    decoder_embed_dim: int,
    # image parameters
    tile_size: int,
    max_num_tiles: int = 4,
    in_channels: int = 3,
) -> Qwen2_5VisionEncoder:
    """
    Build the Qwen2.5 Vision Encoder.

    Args:
        patch_size (int): The size of each patch. Used to divide the tiles into patches.
            E.g. for ``patch_size=40``, a tile of shape (400, 400) will have 10x10 grid of patches
            with shape (40, 40) each.
        num_heads (int): The number of attention heads in each transformer layer.
        clip_embed_dim (int): The dimensionality of each patch embedding in CLIP.
        clip_num_layers (int): The number of transformer layers.
        clip_hidden_states (Optional[List[int]]): The indices of CLIP hidden layers to return
            to return to the encoder projection head. It will return the intermediate results
            of the vision transformer layers which will be concatenated with the CLIP output
            and input into the projection head. For example, ``clip_hidden_states=[0,3]`` will
            return the embeddings before they go through the first and fourth layers.
        num_layers_projection (int): The number of transformer layers in the projection head.
        decoder_embed_dim (int): The dimensionality of the final output embeddings for the decoder.
        tile_size (int): The size of your image tiles, if the image was tile-cropped in advance. Otherwise,
            the size of the input image. In this case, the function will consider your image as a single tile.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.

    Returns:
        Llama3VisionEncoder: Instantiation of Llama 3.2 vision encoder.
    """

    # visual encoder
    visual_encoder = clip_vision_encoder(
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=clip_embed_dim,
        num_layers=clip_num_layers,
        num_heads=num_heads,
        activation=nn.GELU,
        out_indices=clip_hidden_states,
        max_num_tiles=max_num_tiles,
        in_channels=in_channels,
        attn_bias=False,
        output_cls_projection=False,
    )

    # Projection head
    projection_head = qwen2_5_vision_projection_head(
        decoder_embed_dim=decoder_embed_dim,
        clip_embed_dim=clip_embed_dim,
        projection_embed_dim=projection_embed_dim,
    )

    return Qwen2_5VisionEncoder(visual_encoder=visual_encoder, projection_head=projection_head)