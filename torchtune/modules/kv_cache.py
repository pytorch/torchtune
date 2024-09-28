# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torch import nn


class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.size = 0
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.size = 0

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Example:
            >>> cache = KVCache(batch_size=2, max_seq_len=16, num_heads=4, head_dim=32, dtype=torch.bfloat16)
            >>> keys, values = torch.ones((2, 4, 8, 32)), torch.ones((2, 4, 8, 32))
            >>> cache.update(keys, values)
            >>> # now positions 0 through 7 are filled
            >>> cache.size
            >>> 8
            >>> keys, values = torch.ones((2, 4, 1, 32)), torch.ones((2, 4, 1, 32))
            >>> cache.update(keys, values)
            >>> # this will fill at position 8
            >>> cache.size
            >>> 9

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            ValueError: if the sequence length of ``k_val`` is longer than the maximum cache sequence length.
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.
        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        if (self.size + seq_len) > self.k_cache.shape[2]:
            raise ValueError(
                f"The current cache has been setup with a sequence length of {self.k_cache.shape[2]}"
                f", but the cache has reached a sequence length of {(self.size + seq_len)}!"
            )
        cache_pos = torch.arange(self.size, self.size + seq_len, device=k_val.device)
        self.size += seq_len

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, cache_pos] = k_val
        v_out[:, :, cache_pos] = v_val

        return k_out, v_out
