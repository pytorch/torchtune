# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchtune.utils.pooling import get_last_non_masked_token


class TestGetLastNonMaskedToken:
    def test_get_last_non_masked_token_multi_batch(self):
        """
        Tests that the last non-padding tokens are correctly selected for a multi-batch input.
        """
        padding_token_idx = 0
        tokens = torch.tensor([[1, 3, 4, 9], [4, 5, 6, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        expected_output = torch.tensor([9, 6, 1, 0])
        idxs = get_last_non_masked_token(tokens == padding_token_idx)
        torch.testing.assert_close(
            tokens[torch.arange(0, tokens.shape[1]), idxs], expected_output
        )

    def test_get_last_non_masked_token_single_batch(self):
        """
        Tests that the last non-padding tokens are correctly selected for a single-batch input.
        """
        padding_token_idx = 0
        tokens = torch.tensor([[1, 3, 4, 9, 0]])
        expected_output = torch.tensor([9])
        idxs = get_last_non_masked_token(tokens == padding_token_idx)

        torch.testing.assert_close(tokens[0, idxs], expected_output)
