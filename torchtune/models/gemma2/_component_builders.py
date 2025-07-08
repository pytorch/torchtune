# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Optional

import torch
from torch import nn
from torchtune.models.gemma._component_builders import gemma_mlp, lora_gemma_mlp
from torchtune.models.gemma.gemma_norm_embedding import GemmaNormEmbeddings
from torchtune.models.gemma.rms_norm import GemmaRMSNorm

from torchtune.models.gemma2._attention_mask import get_sliding_attention_mask

from torchtune.modules import (
    FrozenNF4Linear,
    RotaryPositionalEmbeddings,
    TiedLinear,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.attention import MultiHeadAttention
from torchtune.modules.common_utils import _register_reparametrize_state_dict_hooks
from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

"""
Component builders for the Gemma2 2B, 9B models and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


class TanhSoftCapping(nn.Module):
    def __init__(
        self,
        capping_value: float,
    ) -> None:
        super().__init__()
        self.capping_value = capping_value

    def forward(self, attn_weights):
        attn_weights = attn_weights / self.capping_value
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.capping_value
        return attn_weights


class Gemma2FinalNorm(nn.Module):
    """
    Combines RMSNorm and SoftCapping
    """

    def __init__(self, capping_value: float, embed_dim: int, eps: float) -> None:
        super().__init__()
        self.capping_value = capping_value
        self.rms_norm = GemmaRMSNorm(embed_dim, eps=eps)
        self.logit_capping = TanhSoftCapping(capping_value)

    def forward(self, x):
        x = self.rms_norm(x)
        x = self.logit_capping(x)
        return x


def gemma2(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    hidden_capping_value: float = 50.0,
    final_capping_value: float = 30.0,
    sliding_window_size: int = 4096,
    query_pre_attn_scalar: Optional[int] = None,
) -> TransformerDecoder:
    """
    Build the decoder associated with the gemma2 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space


    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms Default: 1e-6
        rope_base (int): base for the rotary positional embeddings. Default: 10_000

    Returns:
        TransformerDecoder: Instantiation of gemma model.
    """
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )

    layers = torch.nn.ModuleList()
    for layer_idx in range(num_layers):

        mlp = gemma_mlp(dim=embed_dim, hidden_dim=intermediate_dim)

        # Since `nn.SPDA` doesn't support SoftCapping, soft capping is skipped
        self_att = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            kv_cache=None,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            scale=(query_pre_attn_scalar or head_dim) ** -0.5,
        )
        # Sliding window is applied on half of the layers only
        # Currently returns a Tensor Mask so FlashAttention is not used
        mask_mod = (
            partial(get_sliding_attention_mask, sliding_window_size=sliding_window_size)
            if (layer_idx % 2) == 0
            else None
        )

        layer = TransformerSelfAttentionLayer(
            attn=self_att,
            mlp=mlp,
            sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            sa_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mask_mod=mask_mod,
        )
        layers.append(layer)

    tok_embeddings = GemmaNormEmbeddings(vocab_size, embed_dim)
    output_proj = TiedLinear(tok_embeddings)
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        output=output_proj,
        head_dim=head_dim,
        norm=Gemma2FinalNorm(final_capping_value, embed_dim, eps=norm_eps),
    )
    return model


def lora_gemma2(
    lora_attn_modules: list[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    *,
    # gemma args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    hidden_capping_value: float = 50.0,
    final_capping_value: float = 30.0,
    sliding_window_size: int = 4096,
    query_pre_attn_scalar: Optional[int] = None,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Return a version of Gemma with LoRA applied based on the passed in configuration.
    Note: output projection lora is not supported because it is tied to token embeddings

    Args:
        lora_attn_modules (list[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms Default: 1e-6
        rope_base (int): base for the rotary positional embeddings. Default: 10_000
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        TransformerDecoder: Instantiation of Gemma model with LoRA applied to
        a subset of the attention projections in each layer.
    """

    tok_embeddings = GemmaNormEmbeddings(vocab_size, embed_dim)
    output_proj = TiedLinear(tok_embeddings)
    layers = nn.ModuleList()
    for layer_idx in range(num_layers):
        if apply_lora_to_mlp:
            mlp = lora_gemma_mlp(
                dim=embed_dim,
                hidden_dim=intermediate_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
                quantize_base=quantize_base,
            )
        else:
            mlp = gemma_mlp(
                dim=embed_dim, hidden_dim=intermediate_dim, quantize_base=quantize_base
            )
        self_att = lora_gemma2_self_attention(
            lora_modules=lora_attn_modules,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_base=rope_base,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            # perform sliding window on half of the layers only
            sliding_window_size=sliding_window_size if (layer_idx % 2) == 0 else None,
            softcapping=hidden_capping_value,
            query_pre_attn_scalar=query_pre_attn_scalar,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            quantize_base=quantize_base,
        )
        # Sliding window is applied on half of the layers only
        # Currently returns a Tensor Mask so FlashAttention is not used
        mask_mod = (
            partial(get_sliding_attention_mask, sliding_window_size=sliding_window_size)
            if (layer_idx % 2) == 0
            else None
        )

        layer = TransformerSelfAttentionLayer(
            attn=self_att,
            mlp=mlp,
            sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
            sa_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mlp_scale=GemmaRMSNorm(embed_dim, eps=norm_eps),
            mask_mod=mask_mod,
        )
        layers.append(layer)

    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        output=output_proj,
        head_dim=head_dim,
        norm=Gemma2FinalNorm(final_capping_value, embed_dim, eps=norm_eps),
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to higher precision, and offload to CPU on the fly
        # so as to not increase peak memory
        # TODO this is clowny, figure out a better way to get what precision the rest
        # of the model is in
        _register_reparametrize_state_dict_hooks(
            model, dtype=tok_embeddings.weight.dtype
        )

    return model


def lora_gemma2_self_attention(
    lora_modules: list[LORA_ATTN_MODULES],
    *,
    # MultiHeadAttention args
    embed_dim: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 10_000,
    sliding_window_size: Optional[int] = None,
    softcapping: Optional[float] = 50.0,
    query_pre_attn_scalar: Optional[int],
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> MultiHeadAttention:
    if not lora_modules:
        raise ValueError(
            f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules"
        )

    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
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
            num_heads * head_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "output_proj" in lora_modules
        else (
            nn.Linear(num_heads * head_dim, embed_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(num_heads * head_dim, embed_dim, bias=False)
        )
    )

    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )

    # Since `nn.SPDA` doesn't support SoftCapping, soft capping is skipped
    self_att = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
        scale=(query_pre_attn_scalar or head_dim) ** -0.5,
    )
    return self_att
