# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchtune.training.pooling import get_unmasked_sequence_lengths


class TestGetLastUnmaskedTokenIdx:
    def test_get_last_unmasked_token_idx_multi_batch(self):
        """
        Tests that the last non-padding tokens are correctly selected for a multi-batch input.
        """
        padding_token_idx = 0
        tokens = torch.tensor([[1, 3, 4, 9], [4, 5, 6, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        expected_output = torch.tensor([3, 2, 0, 0])
        idxs = get_unmasked_sequence_lengths(tokens == padding_token_idx)
        torch.testing.assert_close(idxs, expected_output)

    def test_get_last_unmasked_token_idx_single_batch(self):
        """
        Tests that the last non-padding tokens are correctly selected for a single-batch input.
        """
        padding_token_idx = 0
        tokens = torch.tensor([[1, 3, 4, 9, 0]])
        expected_output = torch.tensor([3])
        idxs = get_unmasked_sequence_lengths(tokens == padding_token_idx)

        torch.testing.assert_close(idxs, expected_output)

    def test_get_last_unmasked_token_idx_multi_batch_all_full(self):
        """
        Tests that the last non-padding tokens are correctly selected for multi-batch input,
        where none of the sequences have padding tokens.
        """
        padding_token_idx = 0
        tokens = torch.tensor(
            [[1, 3, 4, 9], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
        expected_output = torch.tensor([3, 3, 3, 3])
        idxs = get_unmasked_sequence_lengths(tokens == padding_token_idx)

        torch.testing.assert_close(idxs, expected_output)

    def test_get_last_unmasked_token_idx_multi_batch_all_empty(self):
        """
        Tests that the last non-padding tokens are correctly selected for multi-batch input,
        where none of the sequences have any non-padding tokens.
        """
        padding_token_idx = 0
        tokens = torch.zeros((4, 4), dtype=torch.long)
        expected_output = torch.tensor([0, 0, 0, 0])
        idxs = get_unmasked_sequence_lengths(tokens == padding_token_idx)

        torch.testing.assert_close(idxs, expected_output)
