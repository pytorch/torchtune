# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from unittest import mock

import pytest
import torch

from torchtune.utils.attention_bias import (
    _get_document_ids_from_seq_lens,
    create_block_causal_mask,
    packed_block_causal_mask,
)


class TestBlockCausalMask:
    @pytest.fixture
    def seq_lens(self):
        return torch.tensor([[2, 3, 1, 0], [2, 2, 2, 0]])

    def test_get_document_ids_from_seq_lens(self, seq_lens):
        actual = _get_document_ids_from_seq_lens(seq_lens)
        expected = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 0, 1, 1, 2, 2]])
        torch.testing.assert_close(actual, expected)

    def test_create_block_causal_mask(self, seq_lens):
        actual = create_block_causal_mask(seq_lens)
        expected = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0],
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
        )
        torch.testing.assert_close(actual, expected)

    @mock.patch("torchtune.utils.attention_bias.torch_version_ge")
    def test_packed_block_causal_mask_sdpa(self, mock_version, seq_lens):
        mock_version.return_value = False
        actual = packed_block_causal_mask(seq_lens, device="cpu")
        expected = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0],
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
        )
        torch.testing.assert_close(actual, expected)
