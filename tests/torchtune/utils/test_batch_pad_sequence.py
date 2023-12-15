# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

from torchtune.utils.batch_pad_sequence import batch_pad_to_longest_seq


class TestBatchPadSequence:
    def test_batch_pad(self):
        """
        Tests that shorter input, label sequences are padded to the max seq len.
        """
        input_padding_idx = 0
        label_padding_idx = -100
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
        inputs, labels = batch_pad_to_longest_seq(
            batch=token_pairs,
            input_padding_idx=input_padding_idx,
            label_padding_idx=label_padding_idx,
        )
        padded_input = inputs[1]
        padded_label = labels[1]
        assert torch.allclose(
            padded_input, torch.tensor([7, input_padding_idx, input_padding_idx])
        )
        assert torch.allclose(
            padded_label, torch.tensor([10, label_padding_idx, label_padding_idx])
        )
