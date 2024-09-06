# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from unittest import mock

import pytest
import torch

from torchtune.modules.attention_utils import (
    _get_document_ids_from_seq_lens,
    _sdpa_or_flex_attention,
    _SUPPORTS_FLEX_ATTENTION,
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

    @mock.patch("torchtune.modules.attention_utils._SUPPORTS_FLEX_ATTENTION", False)
    def test_packed_block_causal_mask_sdpa(self, mock_supports_flex, seq_lens):
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

    @pytest.mark.skipif(
        not _SUPPORTS_FLEX_ATTENTION,
        reason="Please install a nightly build of torch (>=2.5.0) to run this test.",
    )
    def test_packed_block_causal_mask_flex(self, mock_version, seq_lens):
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


class TestSDPAOrFlexAttention:
    @mock.patch("torchtune.modules.attention_utils._SUPPORTS_FLEX_ATTENTION", False)
    @mock.patch("torchtune.modules.attention_utils.flex_attention_compiled")
    @mock.patch("torchtune.modules.attention_utils.nn.scaled_dot_product_attention")
    def test_sdpa_attention(self, mock_sdpa, mock_flex, mock_supports_flex):
        q = torch.ones(2, 3, 4)
        k = torch.ones(2, 3, 4)
        v = torch.ones(2, 3, 4)
        attn_mask = torch.ones(2, 3, 4)
        dropout_p = 0.0
        is_causal = False
        _attention_call = _sdpa_or_flex_attention()
        _ = _attention_call(q, k, v, attn_mask, dropout_p, is_causal)
        mock_sdpa.assert_called_once()
        mock_flex.assert_not_called()

    @pytest.mark.skipif(
        not _SUPPORTS_FLEX_ATTENTION,
        reason="Please install a nightly build of torch (>=2.5.0) to run this test.",
    )
    @mock.patch("torchtune.modules.attention_utils.flex_attention_compiled")
    @mock.patch("torchtune.modules.attention_utils.nn.scaled_dot_product_attention")
    def test_flex_attention(self, mock_sdpa, mock_flex):
        q = torch.ones(2, 3, 4)
        k = torch.ones(2, 3, 4)
        v = torch.ones(2, 3, 4)
        attn_mask = torch.ones(2, 3, 4)
        dropout_p = 0.0
        is_causal = False
        _attention_call = _sdpa_or_flex_attention()
        _ = _attention_call(q, k, v, attn_mask, dropout_p, is_causal)
        mock_sdpa.assert_not_called()
        mock_flex.assert_called_once()
