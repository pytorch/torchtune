# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchtune.utils.pooling import pool_sequence_logits


class TestPooling:
    def test_pool_sequence_logits_multi_batch(self):
        """
        Tests that the last non-padding token logits are pooled correctly for a multi-batch input.
        """
        padding_token_idx = 0
        tokens = torch.tensor([[1, 3, 4, 9], [4, 5, 6, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        logits = torch.tensor(
            [
                [[0.1, 1.3, 1.4], [0.5, 0.6, 0.7], [0.9, 1.1, 1.2], [1.3, 0.5, 1.6]],
                [[0.2, 1.4, 1.5], [0.6, 0.7, 0.8], [1.0, 1.2, 1.3], [1.4, 1.6, 0.7]],
                [[0.3, 1.5, 1.6], [0.1, 1.8, 0.2], [1.1, 1.3, 1.4], [0.5, 1.7, 0.1]],
                [[0.4, 1.6, 1.7], [0.8, 0.9, 1.0], [1.2, 1.4, 1.5], [0.6, 1.8, 0.2]],
            ]
        )
        expected_output = torch.tensor(
            [
                [1.3, 0.5, 1.6],
                [1.0, 1.2, 1.3],
                [0.3, 1.5, 1.6],
                [0.4, 1.6, 1.7],
            ]
        )
        output = pool_sequence_logits(tokens, logits, padding_token_idx)
        torch.testing.assert_close(output, expected_output)

    def test_pool_sequence_logits_single_batch(self):
        """
        Tests that the last non-padding token logits are pooled correctly for a single-batch input.
        """
        padding_token_idx = 0
        tokens = torch.tensor([[1, 3, 4, 9]])
        logits = torch.tensor(
            [
                [[0.1, 1.3, 1.4], [0.5, 0.6, 0.7], [0.9, 1.1, 1.2], [1.3, 0.5, 1.6]],
            ]
        )
        expected_output = torch.tensor(
            [
                [1.3, 0.5, 1.6],
            ]
        )
        output = pool_sequence_logits(
            tokens, logits, padding_token_idx=padding_token_idx
        )
        torch.testing.assert_close(output, expected_output)
