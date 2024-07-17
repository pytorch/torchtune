# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask
from functools import lru_cache


def sample_packing_block_causal_mask(document_ids: Tensor) -> Callable:
    def mask_mod(b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
        padding_mask = (document_ids[b, q_idx] != -1) & (document_ids[b, kv_idx] != -1)
        return causal_mask & document_mask & padding_mask

    return mask_mod

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask
