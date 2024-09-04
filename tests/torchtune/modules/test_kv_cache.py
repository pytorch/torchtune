# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.modules import KVCache

BSZ = 2
MAX_SEQ_LEN = 16
NUM_HEADS = 4
HEAD_DIM = 256
DTYPE = torch.float32


class TestKVCache:
    @pytest.fixture()
    def k_vals_full(self):
        return torch.ones((BSZ, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM))

    @pytest.fixture()
    def v_vals_full(self):
        return torch.ones((BSZ, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM)) * 2

    @pytest.fixture()
    def kv_cache(self):
        return KVCache(BSZ, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM, DTYPE)

    def test_kv_cache_init(self, kv_cache):
        # kv cache should be init with zero
        assert (kv_cache.k_cache == 0).all() and (kv_cache.v_cache == 0).all()

    def test_kv_cache_reset(self, kv_cache, k_vals_full, v_vals_full):
        kv_cache.update(k_vals_full, v_vals_full)
        kv_cache.reset()
        assert (kv_cache.k_cache == 0).all() and (kv_cache.v_cache == 0).all()

    def test_kv_cache_error_when_bsz_exceeded(self, kv_cache, k_vals_full, v_vals_full):
        with pytest.raises(ValueError):
            kv_cache.update(k_vals_full.repeat(4, 1, 1, 1), v_vals_full)

    def test_kv_cache_error_when_seq_len_exceeded(
        self, kv_cache, k_vals_full, v_vals_full
    ):
        with pytest.raises(ValueError):
            kv_cache.update(k_vals_full.repeat(1, 1, 4, 1), v_vals_full)

    def test_kv_cache_error_when_seq_len_exceeded_after_update(
        self, kv_cache, k_vals_full, v_vals_full
    ):
        # test that the cache position is correctly being used to check for seq len exceeded
        # make a valid update filling half the cache
        kv_cache.update(
            k_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
            v_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
        )
        with pytest.raises(
            ValueError,
            match=f"cache has reached a sequence length of {MAX_SEQ_LEN + MAX_SEQ_LEN // 2}",
        ):
            # now an invalid update exceeding the cache
            kv_cache.update(k_vals_full, v_vals_full)

    def test_kv_cache_size_update(self, kv_cache, k_vals_full, v_vals_full):
        # tests that the kv_cache is correctly tracking the cache position

        # make a valid update filling half the cache - like a prefill
        kv_cache.update(
            k_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
            v_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
        )
        assert kv_cache.size == MAX_SEQ_LEN // 2
        # now one update with the next key and value
        kv_cache.update(
            k_vals_full[:, :, (MAX_SEQ_LEN // 2) + 1].unsqueeze(-2),
            v_vals_full[:, :, (MAX_SEQ_LEN // 2) + 1].unsqueeze(-2),
        )
        assert kv_cache.size == (MAX_SEQ_LEN // 2) + 1
