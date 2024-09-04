# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import pytest
import torch
from torchtune.data import (
    left_pad_sequence,
    padded_collate,
    padded_collate_dpo,
    padded_collate_sft,
)


class TestPaddedCollateSFT:
    def test_batch_pad_sequence(self):
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
        padded = padded_collate_sft(
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


class TestLeftPadSequence:
    def test_left_pad_sequence(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6, 7])
        c = torch.tensor([8, 9, 10, 11, 12])
        result = left_pad_sequence([a, b, c], batch_first=True, padding_value=0)
        expected = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7], [8, 9, 10, 11, 12]])
        assert torch.equal(result, expected)


class TestPaddedCollate:
    def test_padded_collate_classifier_labels(self):
        batch = [
            {"tokens": [1, 2, 3], "labels": 1},
            {"tokens": [4, 5], "labels": 2},
            {"tokens": [6, 7, 8, 9], "labels": 3},
        ]
        result = padded_collate(
            batch,
            pad_direction="right",
            keys_to_pad=["tokens"],
            padding_idx=-10,
        )
        expected_tokens = torch.tensor([[1, 2, 3, -10], [4, 5, -10, -10], [6, 7, 8, 9]])
        expected_labels = torch.tensor([1, 2, 3])
        assert torch.equal(result["tokens"], expected_tokens)
        assert torch.equal(result["labels"], expected_labels)

    def test_padded_collate_multiple_keys_to_pad(self):
        batch = [
            {"tokens": [1, 2], "labels_0": [3, 4], "labels_1": 1},
            {"tokens": [5, 6, 7], "labels_0": [8, 9, 10], "labels_1": 2},
        ]
        result = padded_collate(
            batch,
            pad_direction="left",
            keys_to_pad=["tokens", "labels_0"],
            padding_idx={"tokens": 0, "labels_0": -1},
        )
        expected_tokens = torch.tensor([[0, 1, 2], [5, 6, 7]])
        expected_labels_0 = torch.tensor([[-1, 3, 4], [8, 9, 10]])
        expected_labels_1 = torch.tensor([1, 2])
        assert torch.equal(result["tokens"], expected_tokens)
        assert torch.equal(result["labels_0"], expected_labels_0)
        assert torch.equal(result["labels_1"], expected_labels_1)

    def test_value_error_raised_when_empty_keys_to_pad(self):
        batch = [{"labels": [1]}, {"labels": [2]}]
        with pytest.raises(ValueError):
            padded_collate(batch, pad_direction="left", keys_to_pad=[], padding_idx=0)

    def test_value_error_raised_when_mismatched_padding_idx_keys(self):
        batch = [{"tokens": [1, 2], "labels": [1, 1]}]
        with pytest.raises(ValueError):
            padded_collate(
                batch,
                pad_direction="left",
                keys_to_pad=["tokens", "labels"],
                padding_idx={"tokens": 0},
            )

    def test_value_error_raised_when_mismatched_keys_to_pad(self):
        batch = [{"tokens": [1, 2], "labels": [1, 1]}]
        with pytest.raises(ValueError):
            padded_collate(
                batch,
                pad_direction="left",
                keys_to_pad=["tokens", "labels_0"],
                padding_idx={"tokens": 0},
            )

    def test_value_error_raised_when_invalid_pad_direction(self):
        batch = [{"tokens": [1, 2], "labels": [1, 1]}]
        with pytest.raises(ValueError):
            padded_collate(
                batch,
                pad_direction="oogabooga",
                keys_to_pad=["tokens", "labels_0"],
                padding_idx={"tokens": 0},
            )


class TestPaddedCollateDPO:
    def test_dpo_collate(self):
        batch = [
            {
                "chosen_input_ids": [1, 2, 3],
                "chosen_labels": [4, 5, 6],
                "rejected_input_ids": [7, 8],
                "rejected_labels": [9, 10],
            },
            {
                "chosen_input_ids": [11, 12],
                "chosen_labels": [13, 14],
                "rejected_input_ids": [15, 16, 17],
                "rejected_labels": [18, 19, 20],
            },
        ]
        input_ids, labels = padded_collate_dpo(batch, padding_idx=0, ignore_idx=-100)
        expected_input_ids = torch.tensor(
            [[1, 2, 3], [11, 12, 0], [7, 8, 0], [15, 16, 17]]
        )
        expected_labels = torch.tensor(
            [[4, 5, 6], [13, 14, -100], [9, 10, -100], [18, 19, 20]]
        )
        assert torch.equal(input_ids, expected_input_ids)
        assert torch.equal(labels, expected_labels)
