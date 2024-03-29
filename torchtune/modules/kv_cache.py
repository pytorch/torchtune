# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn, Tensor


class KVCache(nn.Module):
    """
    Standalone nn.Module containing a kv-cache to cache past key and values during inference.

    Args:
        max_batch_size (int): maximum batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        n_kv_heads (int): number of kv heads
        head_dim (int): per-attention head embedding dimension
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype
    ):
        super().__init__()
        cache_shape = (max_batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.max_batch_size = max_batch_size

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
