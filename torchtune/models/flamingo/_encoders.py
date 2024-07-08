# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torch import nn

from torchtune.modules import (
    Fp32LayerNorm,
    GroupedQueryAttention,
    TanhGate,
    TransformerSelfAttentionLayer,
)


class FlamingoVisionAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        proj_in: int,
        proj_out: int,
    ) -> None:
        super().__init__()

        mlp_ratio = 4
        hidden_dim = int(mlp_ratio * embed_dim)
        head_dim = embed_dim // num_heads
        num_kv_heads = num_heads

        transformer_layers = []
        for idx in range(1, num_layers + 1):
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

            mlp = FlamingoMLP(
                in_dim=embed_dim,
                hidden_dim=hidden_dim,
                out_dim=embed_dim,
                act_layer=nn.GELU,
            )

            layer = TransformerSelfAttentionLayer(
                attn=attn,
                mlp=mlp,
                attn_norm=Fp32LayerNorm(embed_dim),
                mlp_norm=Fp32LayerNorm(embed_dim),
                attn_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )

            transformer_layers.append(layer)

        self.projection = nn.Linear(proj_in, proj_out)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:

        bsz, n_ims, n_tiles, n_tokens, embed_dim = x

        # transformer
        x = x.view(bsz * n_ims, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            x = transformer_layer(x)

        # concat
        x = x.view(bsz, n_ims, n_tiles, n_tokens, embed_dim)
        x = torch.cat([x, hidden_states], dim=-1)
        x = self.projection(x)
        return x


class FlamingoVisionEncoder(nn.Module):
    def __init__(self, vision_encoder: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.adapter = adapter

    def forward(
        self, x: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x, hidden_states = self.vision_encoder(x, aspect_ratio)
        x = self.adapter(x, hidden_states)
        return x


class FlamingoMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        act_layer: nn.Module,
        dropout=0.0,
    ):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.activation = act_layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x.float()).type_as(x)
        x = self.dropout(x)
        return self.layer2(x)
