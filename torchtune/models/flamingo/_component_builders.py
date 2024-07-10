from typing import Optional, List

from torch import nn

from torchtune.models.llama3._component_builders import llama3_mlp
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.modules.model_fusion import FusionEmbedding, FusionLayer

from torchtune.modules import (
    CausalSelfAttention,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoderLayer,
    TanhGate,
    TransformerCrossAttentionLayer,
    GroupedQueryAttention,
    MMTransformerDecoder,
    TransformerSelfAttentionLayer,
    Fp32LayerNorm
)

from torchtune.models.clip._component_builders import clip_vision_encoder
from torchtune.models.flamingo._encoders import FlamingoVisionAdapter, FlamingoVisionEncoder

from torchtune.modules.feed_forward import MLP

"""
Component builders for the Flamingo model and it's constituant models.
torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``GroupedQueryAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""

def flamingo_vision_encoder(
    tile_size: int,
    patch_size: int,
    num_heads: int,
    embed_dim: int,
    num_layers_clip: int,
    num_layers_adapter: int,
    embed_dim_out: int,
    out_indices: Optional[List[int]] = None,
    max_num_tiles: int = 4,
    in_channels: int = 3,
    ) -> FlamingoVisionEncoder:

    # clip encoder
    clip = clip_vision_encoder(
        tile_size=tile_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=num_layers_clip,
        num_heads=num_heads,
        out_indices=out_indices,
        max_num_tiles=max_num_tiles,
        in_channels=in_channels,
        output_cls_projection=False,
    )

    # Flamingo Vision Adapter
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * embed_dim)
    head_dim = embed_dim // num_heads
    num_kv_heads = num_heads

    transformer_adapter_layers = []
    for _ in range(num_layers_adapter):
        self_attn = GroupedQueryAttention(
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
            default_causal_mask=False,
        )

        mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_dim,
            act_layer=nn.GELU(),
        )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            attn_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
            mlp_norm=Fp32LayerNorm(embed_dim, eps=1e-5),
            attn_scale=TanhGate(),
            mlp_scale=TanhGate(),
        )

        transformer_adapter_layers.append(layer)
    
    transformer_adapter_layers = nn.ModuleList(transformer_adapter_layers)

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    clip_emb_size = embed_dim # output_cls_projection = False
    hidden_emb_size = embed_dim
    num_hidden_out = len(out_indices) if out_indices is not None else 0
    proj_in = clip_emb_size + (hidden_emb_size * num_hidden_out)

    projection = nn.Linear(proj_in, embed_dim_out)

    adapter = FlamingoVisionAdapter(
        layers = transformer_adapter_layers,
        projection = projection,
    )

    return FlamingoVisionEncoder(vision_encoder=clip, adapter=adapter)


def flamingo_text_decoder(
    vocab_size: int,
    num_layers: int,
    fusion_interval: int,
    num_special_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
) -> MMTransformerDecoder:
    """
    Build the decoder associated with the Llama3 model with additonal fused
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
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`.

    Returns:
        MMTransformerDecoder: Instantiation of Flamingo model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    hidden_dim = intermediate_dim or scale_hidden_dim_for_mlp(embed_dim)
    layers = []
    for idx in range(1, num_layers + 1):

        # Self attention layers for text decoder
        rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
        self_attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            attn_dropout=0.0,
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            attn_norm=RMSNorm(dim=embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(dim=embed_dim, eps=1e-5),
        )

        # cross attention layers, mixing text and vision,
        # placed every `fusion_interval` layers
        if idx % fusion_interval == 0:
            attn = GroupedQueryAttention(
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
                default_causal_mask=False,
                attn_dropout=0.0,
            )
            mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                attn_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
                attn_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            fusion_layer = FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer)
            layers.append(fusion_layer)
        else:
            layers.append(decoder_layer)

    layers = nn.ModuleList(layers)
    tok_embeddings = FusionEmbedding(vocab_size, num_special_tokens, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

    return MMTransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layers,
        num_layers=None,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=1e-05),
        output=output_proj,
    )
