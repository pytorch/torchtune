# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

from torchtune.utils.collate import _padded_collate_packed, padded_collate


class TestBatchPadSequence:
    def test_padded_collate(self):
        """
        Tests that shorter input, label sequences are padded to the max seq len.
        """
        padding_idx = -8
        ignore_idx = -9
        token_pairs = [
            {
                "tokens": [1, 2, 3],
                "labels": [4, 5, 6],
            },
            {
                "tokens": [7],
                "labels": [10],
            },
        ]
        padded = padded_collate(
            batch=token_pairs,
            padding_idx=padding_idx,
            ignore_idx=ignore_idx,
        )
        padded_input = padded["tokens"][1]
        padded_label = padded["labels"][1]
        torch.testing.assert_close(
            padded_input, torch.tensor([7, padding_idx, padding_idx])
        )
        torch.testing.assert_close(
            padded_label, torch.tensor([10, ignore_idx, ignore_idx])
        )


class TestSamplePadSequence:
    def test_padded_collate_packed(self):
        padding_idx = -8
        ignore_idx = -9
        pack = {
            "tokens": [2, 5],
            "labels": [3, 7],
            "mask": torch.tensor([[True, False], [True, True]]),
            "input_pos": [0, 1],
        }

        padded = _padded_collate_packed(
            pack, max_seq_len=4, padding_idx=padding_idx, ignore_idx=ignore_idx
        )
        padded_input = padded["tokens"]
        padded_label = padded["labels"]
        padded_mask = padded["mask"]
        padded_input_pos = padded["input_pos"]

        torch.testing.assert_close(
            padded_input, torch.tensor([2, 5, padding_idx, padding_idx])
        )
        torch.testing.assert_close(
            padded_label, torch.tensor([3, 7, ignore_idx, ignore_idx])
        )
        torch.testing.assert_close(
            padded_mask,
            torch.tensor(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ),
        )
        torch.testing.assert_close(padded_input_pos, torch.tensor([0, 1, 2, 3]))
