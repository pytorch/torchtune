# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from functools import partial
from typing import List, Optional

from torch import nn
from torchtune.models.clip._component_builders import (
    clip_mlp,
    clip_vision_encoder,
    lora_clip_attention,
    lora_clip_mlp,
    lora_clip_vision_encoder,
)

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama3_1._component_builders import (
    llama3_mlp,
    lora_llama3_attention,
    lora_llama3_mlp,
)
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.models.llama3_2_vision._encoder import (
    Llama3VisionEncoder,
    Llama3VisionProjectionHead,
)
from torchtune.modules import (
    Fp32LayerNorm,
    MultiHeadAttention,
    RMSNorm,
    TanhGate,
    TransformerCrossAttentionLayer,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torchtune.modules.model_fusion import FusionEmbedding, FusionLayer

from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear


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


def llama3_2_vision_encoder(
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
) -> Llama3VisionEncoder:
    """
    Build the Llama 3.2 vision encoder by combining the CLIP image model with an additional
    projection head fusion module. This includes:
    - Spatial positional encodings
    - CLIP model backbone
    - Projection head on top of CLIP
    - Final projection into token embedding dimension

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

    # clip encoder
    clip = clip_vision_encoder(
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
    projection_head = llama3_2_vision_projection_head(
        num_layers=num_layers_projection,
        num_heads=num_heads,
        decoder_embed_dim=decoder_embed_dim,
        clip_embed_dim=clip_embed_dim,
        num_hidden_inputs=len(clip_hidden_states or []),
    )

    return Llama3VisionEncoder(clip=clip, projection_head=projection_head)


def llama3_2_vision_decoder(
    *,
    vocab_size: int,
    num_layers: int,
    fusion_interval: int,
    num_special_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    encoder_max_seq_len: int,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3 model with additional fused
    cross attention layers. This includes:
    - Token embeddings
    - num_layers number of CausalSelfAttention blocks
    - Fused cross attention layers every fusion_interval number of layers
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        fusion_interval (int): interval number of layers between fusion layers.
        num_special_tokens (int): number of special tokens added for the fusion model.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value.
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`.
        encoder_max_seq_len (int): maximum sequence length the encoder will be run with, as used
            by :func:`~torchtune.modules.KVCache`.
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`.

    Returns:
        TransformerDecoder: Instantiation of Llama 3.2 vision decoder.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = intermediate_dim or scale_hidden_dim_for_mlp(embed_dim)
    layers = []

    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    for idx in range(1, num_layers + 1):

        # Self attention layers for text decoder
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=0.0,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(dim=embed_dim, eps=1e-5),
        )

        # cross attention layers, mixing text and vision,
        # placed every `fusion_interval` layers
        if idx % fusion_interval == 0:
            attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
                k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
                output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
                q_norm=RMSNorm(dim=head_dim, eps=1e-05),
                k_norm=RMSNorm(dim=head_dim, eps=1e-05),
                pos_embeddings=None,
                max_seq_len=encoder_max_seq_len,
                is_causal=False,
                attn_dropout=0.0,
            )
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                ca_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
                ca_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer)
            layers.append(fusion_layer)
        else:
            layers.append(decoder_layer)

    tok_embeddings = FusionEmbedding(vocab_size, num_special_tokens, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=1e-05),
        output=output_proj,
    )


def llama3_2_vision_projection_head(
    *,
    num_layers: int,
    num_heads: int,
    decoder_embed_dim: int,
    clip_embed_dim: int,
    num_hidden_inputs: int,
) -> Llama3VisionProjectionHead:
    """
    Build the Llama 3.2 Vision Projection Head that maps the output of the CLIP encoder
    to the decoder cross attention input.

    Args:
        num_layers (int): number of layers in the projection head.
        num_heads (int): number of heads in the projection head.
        decoder_embed_dim (int): embedding dimension for the decoder.
        clip_embed_dim (int): embedding dimension for the CLIP encoder.
        num_hidden_inputs (int): number of hidden inputs to the projection head.

    Returns:
        Llama3VisionProjectionHead: Instantiation of Llama 3.2 vision projection head.
    """
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * clip_embed_dim)
    head_dim = clip_embed_dim // num_heads
    num_kv_heads = num_heads

    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=clip_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(clip_embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(clip_embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(clip_embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(clip_embed_dim, clip_embed_dim, bias=False),
            pos_embeddings=None,
            attn_dropout=0.0,
            is_causal=False,
        )

        mlp = clip_mlp(
            in_dim=clip_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=clip_embed_dim,
            activation=nn.GELU(),
        )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            mlp_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            sa_scale=TanhGate(),
            mlp_scale=TanhGate(),
        )
        layers.append(layer)

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    proj_in = clip_embed_dim * (num_hidden_inputs + 1)
    return Llama3VisionProjectionHead(
        layers=layers,
        output=nn.Linear(proj_in, decoder_embed_dim),
        num_hidden_inputs=num_hidden_inputs,
    )


# ------------------ LoRA Llama 3.2 Vision ------------------


class LoRATrainable(Enum):
    FULL = "full"
    LORA = "lora"
    FROZEN = "frozen"


def lora_llama3_2_vision_encoder(
    encoder_lora: bool,
    fusion_lora: bool,
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # clip encoder parameters
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
    # LoRA parameters
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> Llama3VisionEncoder:
    """
    Build the Llama 3.2 vision encoder by combining the CLIP image model with an additional
    projection head fusion module. This includes:
    - Spatial positional encodings
    - CLIP model backbone
    - Projection head on top of CLIP
    - Final projection into token embedding dimension

    Args:
        encoder_lora (bool): whether to apply LoRA to the CLIP encoder
        fusion_lora (bool): whether to apply LoRA to the projection head
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
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
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.


    Returns:
        Llama3VisionEncoder: Instantiation of Llama 3.2 vision encoder.
    """
    lora_options = {
        "lora_modules": lora_attn_modules,
        "apply_lora_to_mlp": apply_lora_to_mlp,
        "apply_lora_to_output": apply_lora_to_output,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_dora": use_dora,
        "quantize_base": quantize_base,
    }

    # clip encoder
    clip_options = {
        "tile_size": tile_size,
        "patch_size": patch_size,
        "embed_dim": clip_embed_dim,
        "num_layers": clip_num_layers,
        "num_heads": num_heads,
        "activation": nn.GELU,
        "out_indices": clip_hidden_states,
        "max_num_tiles": max_num_tiles,
        "in_channels": in_channels,
        "attn_bias": False,
        "output_cls_projection": False,
    }
    if encoder_lora:
        clip = lora_clip_vision_encoder(**clip_options, **lora_options)
    else:
        clip = clip_vision_encoder(**clip_options)

    # Projection
    projection_options = {
        "num_layers": num_layers_projection,
        "num_heads": num_heads,
        "decoder_embed_dim": decoder_embed_dim,
        "clip_embed_dim": clip_embed_dim,
        "num_hidden_inputs": len(clip_hidden_states or []),
    }
    if fusion_lora:
        projection_head = lora_llama3_2_vision_projection_head(
            **projection_options, **lora_options
        )
    else:
        projection_head = lora_llama3_2_vision_projection_head(**projection_options)

    encoder = Llama3VisionEncoder(clip=clip, projection_head=projection_head)

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        encoder._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return encoder


def lora_llama3_2_vision_decoder(
    decoder_lora: bool,
    fusion_lora: bool,
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # decoder params
    vocab_size: int,
    num_layers: int,
    fusion_interval: int,
    num_special_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    encoder_max_seq_len: int,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
    # LoRA parameters
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3 model with additional fused
    cross attention layers. This includes:
    - Token embeddings
    - num_layers number of CausalSelfAttention blocks
    - Fused cross attention layers every fusion_interval number of layers
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        decoder_lora (bool): whether to apply LoRA to the language decoder
        fusion_lora (bool): whether to apply LoRA to the projection head
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        fusion_interval (int): interval number of layers between fusion layers.
        num_special_tokens (int): number of special tokens added for the fusion model.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value.
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`.
        encoder_max_seq_len (int): maximum sequence length the encoder will be run with, as used
            by :func:`~torchtune.modules.KVCache`.
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`.
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        TransformerDecoder: Instantiation of Llama 3.2 vision decoder.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = intermediate_dim or scale_hidden_dim_for_mlp(embed_dim)
    layers = []

    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    for idx in range(1, num_layers + 1):

        # Self attention layers for text decoder
        self_attn = lora_llama3_attention(
            lora_modules=lora_attn_modules,
            pos_embeddings=rope,
            head_dim=head_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=0.0,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            quantize_base=quantize_base,
        )
        if apply_lora_to_mlp:
            mlp = lora_llama3_mlp(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
            )
        else:
            mlp = llama3_mlp(
                dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base
            )
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(dim=embed_dim, eps=1e-5),
        )

        # cross attention layers, mixing text and vision,
        # placed every `fusion_interval` layers
        if idx % fusion_interval == 0:
            attn = lora_llama3_attention(
                lora_modules=lora_attn_modules,
                pos_embeddings=None,
                head_dim=head_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                q_norm=RMSNorm(dim=head_dim, eps=1e-05),
                k_norm=RMSNorm(dim=head_dim, eps=1e-05),
                max_seq_len=encoder_max_seq_len,
                is_causal=False,
                attn_dropout=0.0,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
                quantize_base=quantize_base,
            )
            if apply_lora_to_mlp:
                mlp = lora_llama3_mlp(
                    dim=embed_dim,
                    hidden_dim=hidden_dim,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    quantize_base=quantize_base,
                    lora_dropout=lora_dropout,
                    use_dora=use_dora,
                )
            else:
                mlp = llama3_mlp(
                    dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base
                )
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                ca_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
                ca_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer)
            layers.append(fusion_layer)
        else:
            layers.append(decoder_layer)

    tok_embeddings = FusionEmbedding(vocab_size, num_special_tokens, embed_dim)

    # TODO: quantize_base is not applied to final output_proj currently.
    adapter_cls = DoRALinear if use_dora else LoRALinear
    output_proj = (
        adapter_cls(
            embed_dim,
            vocab_size,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        if apply_lora_to_output
        else nn.Linear(embed_dim, vocab_size, bias=False)
    )

    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=1e-05),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return model


def lora_llama3_2_vision_projection_head(
    lora_modules: List[LORA_ATTN_MODULES],
    *,
    # projection head parameters
    num_layers: int,
    num_heads: int,
    decoder_embed_dim: int,
    clip_embed_dim: int,
    num_hidden_inputs: int,
    # LoRA args
    apply_lora_to_mlp: bool,
    apply_lora_to_output: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> Llama3VisionProjectionHead:
    """
    Build the Llama 3.2 Vision Projection Head with LoRA applied to a subset of the layers.

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to. Options are ``{"q_proj", "k_proj", "v_proj",
            "output_proj"}``.
        num_layers (int): number of layers in the projection head.
        num_heads (int): number of heads in the projection head.
        decoder_embed_dim (int): embedding dimension for the decoder.
        clip_embed_dim (int): embedding dimension for the CLIP encoder.
        num_hidden_inputs (int): number of hidden inputs to the projection head.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Whether to use DoRA layers instead of LoRA layers. Default is ``False``.
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.

    Returns:
        Llama3VisionProjectionHead: Instantiation of Llama 3.2 vision projection head.
    """
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * clip_embed_dim)
    head_dim = clip_embed_dim // num_heads
    num_kv_heads = num_heads

    layers = []
    for _ in range(num_layers):
        self_attn = lora_clip_attention(
            lora_modules=lora_modules,
            embed_dim=clip_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            attn_dropout=0.0,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            quantize_base=quantize_base,
        )

        if apply_lora_to_mlp:
            mlp = lora_clip_mlp(
                in_dim=clip_embed_dim,
                hidden_dim=hidden_dim,
                out_dim=clip_embed_dim,
                activation=nn.GELU(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
            )
        else:
            mlp = clip_mlp(
                in_dim=clip_embed_dim,
                hidden_dim=hidden_dim,
                out_dim=clip_embed_dim,
                activation=nn.GELU(),
                quantize_base=quantize_base,
            )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            mlp_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            sa_scale=TanhGate(),
            mlp_scale=TanhGate(),
        )
        layers.append(layer)

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    # TODO: quantize_base is not applied to final output_proj currently.
    proj_in = clip_embed_dim * (num_hidden_inputs + 1)
    adapter_cls = DoRALinear if use_dora else LoRALinear
    output_proj = (
        adapter_cls(
            proj_in,
            decoder_embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_bias=True,
        )
        if apply_lora_to_output
        else nn.Linear(proj_in, decoder_embed_dim)
    )
    return Llama3VisionProjectionHead(
        layers=layers,
        output=output_proj,
        num_hidden_inputs=num_hidden_inputs,
    )
