# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import Tensor


class KVCache(torch.nn.Module):
    """
    Standalone nn.Module containing a kv-cache to cache past key and values during inference.

    Args:
        max_batch_size (int): maximum batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        n_kv_heads (int): number of kv heads
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): datatype of kv-cache entries (default is torch.float32)
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_kv_heads, max_seq_length, head_dim)
        self.k_cache = torch.nn.Parameter(torch.zeros(cache_shape, dtype=dtype))
        self.v_cache = torch.nn.Parameter(torch.zeros(cache_shape, dtype=dtype))

    def update(
        self, input_pos: torch.Size, k_val: Tensor, v_val: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Updates the kv-cache at input_pos with the given k_val and v_val

        Args:
        input_pos (torch.Size): Position / sequence of positions corresponding to entries
            to update.
        k_val (Tensor): new k value
        v_val (Tensor): new v value

        Returns:
            Tuple[Tensor, Tensor]: the k-cache and v-cache
        """
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
