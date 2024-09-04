# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch

from torchtune.data._collate import (
    padded_collate,
    padded_collate_tiled_images_with_cross_attention,
)


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

    def test_padded_collate_tiled_images_with_cross_attention(self):
        batch = [
            {
                "tokens": [1, 2, 1, 3],
                "labels": [4, 5, 6, 7],
                "images": [torch.ones(2, 1, 1, 1), torch.ones(3, 1, 1, 1)],
                "encoder_mask": [torch.ones(4, 5 * 2), torch.ones(4, 5 * 3)],
                "aspect_ratio": [torch.tensor([1, 2]), torch.tensor([1, 2])],
            },
            {
                "tokens": [1, 4],
                "labels": [8, 9],
                "images": [torch.ones(4, 1, 1, 1)],
                "encoder_mask": [torch.ones(2, 5 * 4)],
                "aspect_ratio": [torch.tensor([1, 2])],
            },
        ]
        actual = padded_collate_tiled_images_with_cross_attention(
            batch=batch, padding_idx=0, ignore_idx=-100
        )

        mask_1 = torch.concat([torch.ones(4, 5 * 2), torch.zeros(4, 10)], dim=1)
        mask_2 = torch.concat([torch.ones(4, 5 * 3), torch.zeros(4, 5)], dim=1)
        mask_3 = torch.concat([torch.ones(2, 5 * 4), torch.zeros(2, 20)], dim=0)
        sample_1 = torch.stack([mask_1, mask_2])
        sample_2 = torch.stack([mask_3, torch.zeros(4, 20)])
        expected_mask = torch.stack([sample_1, sample_2])

        expected = {
            "tokens": torch.tensor([[1, 2, 1, 3], [1, 4, 0, 0]]),
            "labels": torch.tensor([[4, 5, 6, 7], [8, 9, -100, -100]]),
            "images": torch.tensor(
                [
                    [
                        [[[[1.0]]], [[[1.0]]], [[[0.0]]], [[[0.0]]]],
                        [[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]]],
                    ],
                    [
                        [[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]],
                        [[[[0.0]]], [[[0.0]]], [[[0.0]]], [[[0.0]]]],
                    ],
                ]
            ),
            "encoder_mask": expected_mask,
            "aspect_ratio": torch.tensor([[[1, 2], [1, 2]], [[1, 2], [1, 1]]]),
        }

        for k in expected:
            torch.testing.assert_close(actual[k], expected[k])
