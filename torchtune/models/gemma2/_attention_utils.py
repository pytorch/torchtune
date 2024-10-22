# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch

from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION

if _SUPPORTS_FLEX_ATTENTION:
    from functools import lru_cache

    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask,
        flex_attention,
    )

    # flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

    @lru_cache
    def create_block_mask_cached(score_mod, b, h, m, n, device="cuda"):
        block_mask = create_block_mask(score_mod, b, h, m, n, device=device)
        return block_mask

    # We cannot do nested compile, but flex attention only has perf benefits
    # when compiled. To insulate it from the compiler, we wrap it with
    # compiler.disable so that it can be used regardless of whether the model
    # is compiled or not, and flex attention always remains compiled.
    @torch.compiler.disable(recursive=False)
    def compile_friendly_flex_attention_with_score_and_block(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
        score_mod: Any,
    ) -> torch.Tensor:
        """
        Flex attention does not seem to work with my A6000 with the default options.
        Using proposed options here: https://github.com/pytorch/pytorch/issues/133254
        """
        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            # kernel_options={
            #     "BLOCK_M": 64,
            #     "BLOCK_N": 64,
            #     "BLOCK_M1": 32,
            #     "BLOCK_N1": 64,
            #     "BLOCK_M2": 64,
            #     "BLOCK_N2": 32,
            # },
        )


def flex_causal_sliding_window(sliding_window_size):
    def sliding_window_causal_mask(b, h, q_idx, kv_idx):
        """Causal mask and sliding window as proposed here:
        https://github.com/pytorch-labs/attention-gym/blob/main/examples/flex_attn.ipynb
        """
        causal_mask = q_idx >= kv_idx
        if sliding_window_size is None:
            # if no sliding window return causal mask
            return causal_mask
        else:
            windowed_mask = q_idx - kv_idx <= sliding_window_size

            return causal_mask & windowed_mask

    return sliding_window_causal_mask


def flex_tanh_soft_capping_with_scaling(softcapping, query_pre_attn_scalar):
    def tanh_soft_capping_with_scaling(score, b, h, q_idx, kv_idx):
        """
        This handle both simple tanh soft capping and custom scaling
        """
        if query_pre_attn_scalar is None:
            # usual scaling included in FlexAttention
            # TODO: could be made faster with approximate tanh ?
            # https://github.com/pytorch-labs/attention-gym/blob/f7c93ded4abf9fd8d7dc9d8bcbf57e420b891e2d/examples/flex_attn.ipynb#L733
            score = score / softcapping
            score = torch.tanh(score)
            return score * softcapping
        else:
            score = score / softcapping * query_pre_attn_scalar**-0.5
            score = torch.tanh(score)
            return score * softcapping

    return tanh_soft_capping_with_scaling
