# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

from torchtune.modules.rlhf import left_padded_collate


class TestLeftPaddedCollate:
    def test_left_padded_collate(self):
        """
        Tests that input sequences are left-padded to the max seq len.
        """
        padding_idx = -8
        tokens = [
            {
                "tokens": [
                    1,
                    2,
                ],
            },
            {
                "tokens": [3],
            },
            {
                "tokens": [4, 5, 6, 7],
            },
        ]
        padded_tokens = left_padded_collate(batch=tokens, padding_idx=padding_idx)

        expected_padded_tokens = torch.tensor(
            [
                [padding_idx, padding_idx, 1, 2],
                [padding_idx, padding_idx, padding_idx, 3],
                [4, 5, 6, 7],
            ]
        )
        torch.testing.assert_close(padded_tokens, expected_padded_tokens)
