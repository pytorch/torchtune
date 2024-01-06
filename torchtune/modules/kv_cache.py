# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn, Tensor


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
        cache_shape = (max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.k_cache = nn.Parameter(torch.zeros(cache_shape, dtype=dtype))
        self.v_cache = nn.Parameter(torch.zeros(cache_shape, dtype=dtype))
        self.max_batch_size = max_batch_size

    def update(
        self, bsz: int, seq_len: int, curr_pos: int, k_val: Tensor, v_val: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Updates the kv-cache at curr_pos with the given k_val and v_val.

        Args:
            bsz (int): Batch size.
            seq_len (int): Sequence length.
            curr_pos (int): Current position in sequence.
            k_val (Tensor): New k value.
            v_val (Tensor): New v value.

        Raises:
            ValueError: if bsz is greater than the ``max_batch_size`` supported by the model

        Returns:
            Tuple[Tensor, Tensor]: the key-cache and value-cache
        """
        if bsz > self.max_batch_size:
            raise ValueError(
                f"Batch size {bsz} greater than max batch size {self.max_batch_size}"
            )

        self.k_cache[:bsz, curr_pos : curr_pos + seq_len] = k_val
        self.v_cache[:bsz, curr_pos : curr_pos + seq_len] = v_val
        return (
            self.k_cache[:bsz, : curr_pos + seq_len],
            self.v_cache[:bsz, : curr_pos + seq_len],
        )
