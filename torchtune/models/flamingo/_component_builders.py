from typing import Optional, List

from torch import nn

from torchtune.models.llama3._component_builders import llama3_mlp
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.clip import clip_vision_encoder, clip_mlp
from torchtune.models.flamingo._encoder import FlamingoProjectionHead, FlamingoEncoder

from torchtune.modules.model_fusion import FusionEmbedding, FusionLayer
from torchtune.modules import (
    RMSNorm,
    RotaryPositionalEmbeddings,
    TanhGate,
    TransformerCrossAttentionLayer,
    MultiHeadAttention,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    Fp32LayerNorm
)



"""
Component builders for the Flamingo model and its constituent models.
torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``GroupedQueryAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""

def flamingo_vision_encoder(
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
    ) -> FlamingoEncoder:
    """
    Build the Flamingo encoder by combining the CLIP image model with an additional
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
        FlamingoEncoder: Instantiation of Flamingo encoder.
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
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * clip_embed_dim)
    head_dim = clip_embed_dim // num_heads
    num_kv_heads = num_heads

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

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    num_hidden_inputs = len(clip_hidden_states) if clip_hidden_states is not None else 0
    proj_in = clip_embed_dim * (num_hidden_inputs + 1)
    projection_head = FlamingoProjectionHead(
        layer=layer,
        num_layers=num_layers_projection,
        output=nn.Linear(proj_in, decoder_embed_dim),
        num_hidden_inputs=num_hidden_inputs
    )

    return FlamingoEncoder(clip=clip, projection_head=projection_head)


def flamingo_decoder(
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
        TransformerDecoder: Instantiation of Flamingo decoder.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = intermediate_dim or scale_hidden_dim_for_mlp(embed_dim)
    layers = []
    for idx in range(1, num_layers + 1):

        # Self attention layers for text decoder
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
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
