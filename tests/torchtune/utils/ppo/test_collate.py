# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

from torchtune.utils.ppo import left_padded_collate


class TestBatchPadSequence:
    def test_left_padded_collate(self):
        """
        Tests that shorter input, label sequences are left-padded to the max seq len.
        """
        padding_idx = -8
        max_seq_len = 3
        tokens = [
            {
                "tokens": [1, 2, 3],
            },
            {
                "tokens": [7],
            },
        ]
        padded_tokens = left_padded_collate(
            batch=tokens, padding_idx=padding_idx, max_seq_len=3
        )

        expected_padded_tokens = torch.tensor(
            [
                [1, 2, 3],
                [padding_idx, padding_idx, 7],
            ]
        )
        torch.testing.assert_close(padded_tokens, expected_padded_tokens)
