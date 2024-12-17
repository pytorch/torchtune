# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
from torchtune.modules.kv_cache import KVCache as TuneKVCache


class KVCache(TuneKVCache):
    """
    NOTE: torch.export.export() friendly KVCache implementation modified from KVCache:
    https://github.com/pytorch/torchtune/blob/main/torchtune/modules/kv_cache.py
    Major differences:
    - Changed += to .add_ to avoid mutating module attributes.
    - Added clone() method.
    - Takes a new `transpose_cache` argument to be able to store transposed kv values.

    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
        transpose_cache (bool): whether we transpose(1, 2) for kv cache.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        transpose_cache: bool = True,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        self.transpose_cache = transpose_cache
        self.max_seq_len = max_seq_len
        if self.transpose_cache:
            cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        else:
            cache_shape = (batch_size, max_seq_len, num_kv_heads, head_dim)

        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.arange(0, self.max_seq_len), persistent=False
        )
        self.batch_size = batch_size

    def update(
        self, k_val: torch.Tensor, v_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Example:
            >>> cache = KVCache(batch_size=2, max_seq_len=16, num_kv_heads=4, head_dim=32, dtype=torch.bfloat16)
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
            AssertionError: if the sequence length of ``k_val`` is longer than the maximum cache sequence length.
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.
        """
        if self.transpose_cache:
            bsz, _, seq_len, _ = k_val.shape
        else:
            bsz, seq_len, _, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[0] + seq_len) <= self.max_seq_len

        k_out = self.k_cache
        v_out = self.v_cache

        if self.transpose_cache:
            k_out[:, :, self.cache_pos[:seq_len]] = k_val
            v_out[:, :, self.cache_pos[:seq_len]] = v_val
        else:
            k_out[:, self.cache_pos[:seq_len]] = k_val
            v_out[:, self.cache_pos[:seq_len]] = v_val

        # forward cache_pos seq_len positions along
        # cache_pos starts at (0, 1, 2, 3, 4, 5, ...)
        # an update of seq_len = 5 tokens brings it to
        # (5, 6, 7, 8, 9, ...)
        # this allows us to track the current position in the cache
        # after the last update in a compile-friendly way without any dynamism
        # e.g. relying on an int size tracker, or re-creating cache_pos every time
        self.cache_pos.add_(seq_len)

        return k_out, v_out

    def clone(self) -> "KVCache":
        """Create a clone of the KVCache."""
        if self.transpose_cache:
            num_kv_heads = self.k_cache.shape[1]
        else:
            num_kv_heads = self.k_cache.shape[2]
        clone = KVCache(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            num_kv_heads=num_kv_heads,
            head_dim=self.k_cache.shape[3],
            dtype=self.k_cache.dtype,
            transpose_cache=self.transpose_cache,
        )
        clone.k_cache.copy_(self.k_cache)
        clone.v_cache.copy_(self.v_cache)
        clone.cache_pos.copy_(self.cache_pos)
        return clone
