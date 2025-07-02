# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtune.models.gemma2._attention_mask import get_sliding_attention_mask


class TestGetSlidingAttentionMask:
    @pytest.fixture
    def basic_params(self):
        return {"bsz": 2, "seq_len": 4, "sliding_window_size": 2, "device": None}

    def test_get_sliding_attention_mask(self, basic_params):
        """Test that when mask is None, a causal mask is created and sliding window is applied."""
        bsz = 2
        seq_len = 4
        sliding_window_size = 2
        mask = get_sliding_attention_mask(
            mask=None,
            sliding_window_size=basic_params["sliding_window_size"],
            bsz=basic_params["bsz"],
            seq_len=basic_params["seq_len"],
            device=basic_params["device"],
        )

        assert mask.shape == (
            basic_params["bsz"],
            basic_params["seq_len"],
            basic_params["seq_len"],
        )
        assert mask.dtype == torch.bool

        # Check that the mask has the expected sliding window pattern
        # True positions can be attended to, False positions are masked
        expected_pattern = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [False, True, True, False],
                [False, False, True, True],
            ],
            dtype=torch.bool,
        )

        # Check first batch element
        torch.testing.assert_close(mask[0], expected_pattern)
        # All batch elements should be identical
        torch.testing.assert_close(mask[0], mask[1])

    def test_get_sliding_attention_mask_different_window_sizes(self):
        """Test sliding window with different window sizes."""
        bsz, seq_len = 1, 5

        # Test window size 1 (only current position)
        mask = get_sliding_attention_mask(
            mask=None,
            sliding_window_size=1,
            bsz=bsz,
            seq_len=seq_len,
            device=None,
        )

        expected_window_1 = torch.tensor(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ],
            dtype=torch.bool,
        )

        torch.testing.assert_close(mask[0], expected_window_1)

        # Test window size 3
        mask = get_sliding_attention_mask(
            mask=None,
            sliding_window_size=3,
            bsz=bsz,
            seq_len=seq_len,
            device=None,
        )

        expected_window_3 = torch.tensor(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
            ],
            dtype=torch.bool,
        )

        torch.testing.assert_close(mask[0], expected_window_3)

    def test_get_sliding_attention_mask_large_window(self):
        """Test sliding window larger than sequence length."""
        bsz, seq_len = 1, 3
        sliding_window_size = 5  # Larger than seq_len

        mask = get_sliding_attention_mask(
            mask=None,
            sliding_window_size=sliding_window_size,
            bsz=bsz,
            seq_len=seq_len,
            device=None,
        )

        # Should behave like a regular causal mask when window is larger than seq_len
        expected_causal = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=torch.bool,
        )

        torch.testing.assert_close(mask[0], expected_causal)
