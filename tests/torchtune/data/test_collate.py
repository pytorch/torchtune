# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from unittest import mock

import pytest
import torch

from torchtune.data._collate import padded_collate, padded_collate_packed
from torchtune.utils import torch_version_ge


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
            device="cpu",
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

    @mock.patch("torchtune.utils.attention_bias.torch_version_ge")
    def test_padded_collate_packed_sdpa(self, mock_version):
        mock_version.return_value = False
        token_pairs = [
            {
                "tokens": torch.tensor([1, 2, 3, 4, 5, 6]),
                "labels": torch.tensor([7, 8, 9, 10, 11, 12]),
                "input_pos": torch.tensor([0, 1, 2, 0, 1, 0]),
                "seq_lens": torch.tensor([3, 2, 1]),
            },
            {
                "tokens": torch.tensor([13, 14, 15, 16, 17, 18]),
                "labels": torch.tensor([19, 20, 21, 22, 23, 24]),
                "input_pos": torch.tensor([0, 1, 0, 1, 0, 1]),
                "seq_lens": torch.tensor([2, 2, 2]),
            },
        ]
        collated = padded_collate_packed(
            batch=token_pairs,
            device="cpu",
        )
        torch.testing.assert_close(
            collated["tokens"],
            torch.tensor([[1, 2, 3, 4, 5, 6], [13, 14, 15, 16, 17, 18]]),
        )
        torch.testing.assert_close(
            collated["labels"],
            torch.tensor([[7, 8, 9, 10, 11, 12], [19, 20, 21, 22, 23, 24]]),
        )
        torch.testing.assert_close(
            collated["input_pos"],
            torch.tensor([[0, 1, 2, 0, 1, 0], [0, 1, 0, 1, 0, 1]]),
        )
        torch.testing.assert_close(
            collated["mask"],
            torch.tensor(
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ],
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 1],
                    ],
                ],
                dtype=torch.bool,
            ),
        )

    @pytest.mark.skipif(
        not torch_version_ge("2.5.0"),
        reason="Please install a nightly build of torch to run this test.",
    )
    def test_padded_collate_packed_flex(self):
        # create_block_mask requires that seq_len be divisible by 128, the default block size.
        # see https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L636
        batch = [
            {
                "tokens": torch.ones(128, dtype=torch.long),
                "labels": torch.ones(128, dtype=torch.long),
                "input_pos": torch.zeros(128, dtype=torch.long),
                "seq_lens": torch.ones(64, dtype=torch.long) * 2,
            },
            {
                "tokens": torch.ones(128, dtype=torch.long),
                "labels": torch.ones(128, dtype=torch.long),
                "input_pos": torch.zeros(128, dtype=torch.long),
                "seq_lens": torch.ones(32, dtype=torch.long) * 4,
            },
        ]
        collated = padded_collate_packed(
            batch=batch,
            device="cpu",
        )
        torch.testing.assert_close(
            collated["tokens"], torch.stack([torch.ones(128), torch.ones(128)])
        )
        torch.testing.assert_close(
            collated["labels"], torch.stack([torch.ones(128), torch.ones(128)])
        )
        torch.testing.assert_close(
            collated["input_pos"], torch.stack([torch.zeros(128), torch.zeros(128)])
        )
        torch.testing.assert_close(
            collated["mask"],
            torch.tensor([[[[1]]], [[[1]]]], dtype=torch.long),
        )
