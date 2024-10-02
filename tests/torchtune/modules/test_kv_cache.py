# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch._dynamo.testing
from torchtune.modules import KVCache

BSZ = 2
MAX_SEQ_LEN = 16
NUM_HEADS = 4
HEAD_DIM = 256
DTYPE = torch.float32


class TestKVCache:
    @pytest.fixture()
    def k_vals_full(self):
        return (
            torch.tril(torch.ones(MAX_SEQ_LEN, HEAD_DIM))[
                None,
                None,
                :,
                :,
            ]
            .repeat(BSZ, NUM_HEADS, 1, 1)
            .to(DTYPE)
        )

    @pytest.fixture()
    def v_vals_full(self):
        return (
            torch.tril(torch.ones(MAX_SEQ_LEN, HEAD_DIM))[
                None,
                None,
                :,
                :,
            ].repeat(BSZ, NUM_HEADS, 1, 1)
            * 2
        ).to(DTYPE)

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
        assert kv_cache.size == 0

    def test_kv_cache_error_when_bsz_exceeded(self, kv_cache, k_vals_full, v_vals_full):
        with pytest.raises(ValueError):
            kv_cache.update(k_vals_full.repeat(4, 1, 1, 1), v_vals_full)

    def test_kv_cache_error_when_seq_len_exceeded(
        self, kv_cache, k_vals_full, v_vals_full
    ):
        with pytest.raises(AssertionError):
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
            AssertionError,
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

    def test_kv_cache_single_update(self, kv_cache, k_vals_full, v_vals_full):
        # tests that the kv_cache is correctly returning the updated cache values
        # after a single cache update

        # make a valid update filling half the cache - like a prefill
        k_out, v_out = kv_cache.update(
            k_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
            v_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
        )

        expected_k_out = torch.zeros_like(k_vals_full)
        expected_v_out = torch.zeros_like(v_vals_full)

        expected_k_out[:, :, torch.arange(0, (MAX_SEQ_LEN // 2))] = k_vals_full[
            :, :, : (MAX_SEQ_LEN // 2)
        ]
        expected_v_out[:, :, torch.arange(0, (MAX_SEQ_LEN // 2))] = v_vals_full[
            :, :, : (MAX_SEQ_LEN // 2)
        ]

        assert torch.equal(expected_k_out, k_out)
        assert torch.equal(expected_v_out, v_out)

    def test_kv_cache_multiple_updates(self, kv_cache, k_vals_full, v_vals_full):
        # tests that the kv_cache is correctly returning the updated cache values
        # after a single cache update, followed by another cache update with seq_len=1

        # make an update filling half the cache - like a prefill
        # fills position 0 through to (MAX_SEQ_LEN // 2) - 1
        kv_cache.update(
            k_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
            v_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
        )

        # make an update for one more token, which is the value at
        # (MAX_SEQ_LEN // 2)
        k_out, v_out = kv_cache.update(
            k_vals_full[:, :, (MAX_SEQ_LEN // 2)].unsqueeze(2),
            v_vals_full[:, :, (MAX_SEQ_LEN // 2)].unsqueeze(2),
        )

        expected_k_out = torch.zeros_like(k_vals_full)
        expected_v_out = torch.zeros_like(v_vals_full)

        # cache should be incremented by one position
        expected_k_out[:, :, torch.arange(0, ((MAX_SEQ_LEN // 2) + 1))] = k_vals_full[
            :, :, : ((MAX_SEQ_LEN // 2) + 1)
        ]
        expected_v_out[:, :, torch.arange(0, ((MAX_SEQ_LEN // 2) + 1))] = v_vals_full[
            :, :, : ((MAX_SEQ_LEN // 2) + 1)
        ]

        assert torch.equal(expected_k_out, k_out)
        assert torch.equal(expected_v_out, v_out)

    def test_kv_cache_no_recompiles(self, kv_cache, k_vals_full, v_vals_full):
        def fn(k_val, v_val):
            return kv_cache.update(k_val, v_val)

        cnts = torch._dynamo.testing.CompileCounter()
        # this effectively does torch.compile(fn)
        fn = torch._dynamo.optimize(cnts, nopython=True)(fn)

        # make an update filling half the cache - like a prefill
        # fills position 0 through to (MAX_SEQ_LEN // 2) - 1
        kv_cache.update(
            k_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
            v_vals_full[:, :, : (MAX_SEQ_LEN // 2)],
        )

        # now make successive updates for one token position at a time
        # and ensure there are no recompiles
        for i in range(MAX_SEQ_LEN // 2):
            fn(
                k_vals_full[:, :, (MAX_SEQ_LEN // 2) + i].unsqueeze(2),
                v_vals_full[:, :, (MAX_SEQ_LEN // 2) + i].unsqueeze(2),
            )

        assert cnts.frame_count == 1
