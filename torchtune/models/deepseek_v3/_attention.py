# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Optional
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules import RMSNorm
from torchtune.models.deepseek_v3 import DeepSeekV3LatentLinear


class DeepSeekV3Attention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 qk_rope_head_dim: int,
                 v_head_dim: int,
                 qk_nope_head_dim: int,
                 q_head_dim: int,
                 q_proj: nn.Module,
                 kv_proj: DeepSeekV3LatentLinear,
                 output_proj: nn.Module,
                 kv_norm: nn.Module,
                 pos_embeddings: Optional[nn.Module] = None,
                 q_norm: Optional[nn.Module] = None,
                #  kv_cache: Optional[KVCache] = None,
                 max_seq_len: int = 4096,
                 is_causal: bool = True,
                 attn_dropout: float = 0.0,):

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.q_head_dim = q_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal

        # Set layers
        # self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.kv_proj = kv_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.kv_norm = kv_norm
        self.pos_embeddings = pos_embeddings
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[_MaskType] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # q is sometimes decomposed into A/B
        # kv is *always* decomposed

        # when q is decomposed the norm is applied but
        # not otherwise - in this case the norm
        # should be applied after q a proj and before q b proj

        # for kv decomposition pos embeddings need to be extracted before
        # projecting back up

        b, s_x, _ = x.shape
        q = self.q_proj(x)
        q = q.view(b, s_x, self.num_heads, self.q_head_dim)
        q = q.transpose(1, 2)

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        kv, k_pe = self.kv_proj(x)
        kv = kv.view(b, s_x, self.num_kv_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv = kv.transpose(1, 2)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_pe = self.pos_embeddings(q_pe, input_pos=input_pos)
        k_pe = self.pos_embeddings(k_pe, input_pos=input_pos)

        query_states = q_pe.new_empty(b, self.num_heads, s_x, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(b, self.num_heads, s_x, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        output = self._attention_call(
            query_states,
            key_states,
            value_states,
            mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.kv_cache is None and mask is None and self.is_causal,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(b, s_x, -1)
        return self.output_proj(output)
