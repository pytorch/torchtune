# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Callable, List, Optional

import torch
from torch import nn
from torchtune.models.clip._position_embeddings import (
    TiledTokenPositionalEmbedding,
    TilePositionalEmbedding,
    TokenPositionalEmbedding,
)

from torchtune.modules import (
    FeedForward,
    Fp32LayerNorm,
    FrozenNF4Linear,
    MultiHeadAttention,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

from torchtune.modules.vision_transformer import CLSProjection, VisionTransformer


def clip_vision_encoder(
    tile_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    activation: Callable = nn.SiLU,
    cls_output_dim: int = 512,
    attn_bias: bool = True,
    out_indices: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    max_num_tiles: int = 4,
    in_channels: int = 3,
    intermediate_act: torch.nn.Module = torch.nn.SiLU(),
) -> VisionTransformer:
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
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        intermediate_act (torch.nn.Module): The activation function used in the intermediate layers in the transformer encoder.

    Returns:
        A `VisionTransformer` object.

    Raises:
        AssertionError: If ``embed_dim`` is not divisible by ``num_heads``.
    """
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    cls_projection = (
        CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim)
        if output_cls_projection
        else None
    )

    # transformer layer
    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=embed_dim // num_heads,
        q_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        k_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        v_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=attn_bias),
        pos_embeddings=None,
        attn_dropout=0.0,
        is_causal=False,
    )
    mlp = clip_mlp(
        in_dim=embed_dim,
        hidden_dim=4 * embed_dim,
        out_dim=embed_dim,
        activation=activation(),
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
    if max_num_tiles == 1:
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(
            embed_dim=embed_dim, patch_size=patch_size, tile_size=tile_size
        )
    else:
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


def clip_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    activation: nn.Module,
    quantize_base: bool = False,
) -> FeedForward:
    """
    Build the MLP layer associated with the clip model.
    """
    gate_proj = (
        nn.Linear(in_dim, hidden_dim)
        if not quantize_base
        else FrozenNF4Linear(in_dim, hidden_dim, bias=True)
    )
    down_proj = (
        nn.Linear(hidden_dim, out_dim)
        if not quantize_base
        else FrozenNF4Linear(hidden_dim, out_dim, bias=True)
    )
    return FeedForward(
        gate_proj=gate_proj, down_proj=down_proj, up_proj=None, activation=activation
    )


# ------------------ LoRA CLIP ------------------


def lora_clip_vision_encoder(
    lora_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # clip encoder parameters
    tile_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    num_heads: int,
    activation: Callable = nn.SiLU,
    cls_output_dim: int = 512,
    attn_bias: bool = True,
    out_indices: Optional[List[int]] = None,
    output_cls_projection: bool = False,
    max_num_tiles: int = 4,
    in_channels: int = 3,
    intermediate_act: torch.nn.Module = torch.nn.SiLU(),
    # LoRA parameters
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> VisionTransformer:
    """
    Build a LoRA implementation of the CLIP vision encoder.

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
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
        out_indices (Optional[List[int]]): The indices of hidden layers to return.
            If provided, it will return the intermediate results of the transformer layers
            before they go through a next layer. For example, ``out_indices=[0,3]`` will
            return the tokens before they go through the first and fourth layers.
        output_cls_projection (bool): If True, only the CLS token projection will be outputted,
            instead of all tokens. Defaults to False.
        max_num_tiles (int): The maximum number of tiles that can be processed. This is used to
            determine the size of the positional embeddings.
        in_channels (int): The number of image input channels.
        intermediate_act (torch.nn.Module): The activation function used in the intermediate layers in the transformer encoder.
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.


    Returns:
        VisionTransformer: Instantiation of VisionTransformer model.
    """
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

    # TODO: add support for quantizing and LoRA for the final output projection
    cls_projection = (
        CLSProjection(embed_dim=embed_dim, cls_output_dim=cls_output_dim)
        if output_cls_projection
        else None
    )

    # transformer layer
    self_attn = lora_clip_attention(
        lora_modules=lora_modules,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=embed_dim // num_heads,
        attn_dropout=0.0,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
    if apply_lora_to_mlp:
        mlp = lora_clip_mlp(
            in_dim=embed_dim,
            hidden_dim=4 * embed_dim,
            out_dim=embed_dim,
            activation=activation(),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            quantize_base=quantize_base,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
        )
    else:
        mlp = clip_mlp(
            in_dim=embed_dim,
            hidden_dim=4 * embed_dim,
            out_dim=embed_dim,
            activation=activation(),
            quantize_base=quantize_base,
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
    if max_num_tiles == 1:
        pre_tile_pos_embed = None
        post_tile_pos_embed = None
        token_pos_embedding = TokenPositionalEmbedding(
            embed_dim=embed_dim, patch_size=patch_size, tile_size=tile_size
        )
    else:
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

    model = VisionTransformer(
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

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return model


def lora_clip_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # MultiHeadAttention args
    embed_dim: int,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    attn_dropout: float = 0.0,
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
        head_dim (int): dimension of each head in the multihead attention. Usually
            computed as ``embed_dim // num_heads``.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.

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

    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=None,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_clip_mlp(
    *,
    in_dim: int,
    out_dim: int,
    hidden_dim: int,
    activation: nn.Module,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> FeedForward:
    """
    Build the MLP layer with LoRA applied to the gate and down projections.
    """
    adapter_cls = DoRALinear if use_dora else LoRALinear
    gate_proj = adapter_cls(
        in_dim=in_dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
        use_bias=True,
    )
    down_proj = adapter_cls(
        in_dim=hidden_dim,
        out_dim=out_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
        use_bias=True,
    )
    return FeedForward(
        gate_proj=gate_proj, down_proj=down_proj, up_proj=None, activation=activation
    )
