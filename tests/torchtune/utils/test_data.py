# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

from torchtune.utils.data import padded_collate


class TestBatchPadSequence:
    def test_padded_collate(self):
        """
        Tests that shorter input, label sequences are padded to the max seq len.
        """
        padding_idx = -8
        ignore_idx = -9
        token_pairs = [
            ([1, 2, 3], [4, 5, 6]),
            (
                [
                    7,
                ],
                [
                    10,
                ],
            ),
        ]
        inputs, labels = padded_collate(
            batch=token_pairs,
            padding_idx=padding_idx,
            ignore_idx=ignore_idx,
        )
        padded_input = inputs[1]
        padded_label = labels[1]
        assert torch.allclose(padded_input, torch.tensor([7, padding_idx, padding_idx]))
        assert torch.allclose(padded_label, torch.tensor([10, ignore_idx, ignore_idx]))
