# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from unittest import mock

import pytest
import torch
from tests.test_utils import gpu_test

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
        return [torch.tensor([2, 3, 1]), torch.tensor([2, 2, 2, 0])]

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
    def test_packed_block_causal_mask_sdpa(self, seq_lens):
        actual = packed_block_causal_mask(seq_lens)
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
    @gpu_test(gpu_count=1)
    def test_packed_block_causal_mask_flex(self):
        # create_block_mask requires that seq_len be divisible by 128, the default block size.
        # see https://github.com/pytorch/pytorch/blob/3bf6be457d40034aa4b603b7ea1b8977051221ed/torch/nn/attention/flex_attention.py#L792  # noqa
        actual = packed_block_causal_mask(
            [torch.tensor([64, 64]), torch.tensor([64, 64])]
        )
        expected = torch.tensor([[[[1]]], [[[1]]]], device="cuda:0", dtype=torch.int32)
        torch.testing.assert_close(actual.to_dense(), expected)


class TestSDPAOrFlexAttention:
    @pytest.mark.skipif(
        not _SUPPORTS_FLEX_ATTENTION,
        reason="Please install a nightly build of torch (>=2.5.0) to run this test.",
    )
    @mock.patch("torchtune.modules.attention_utils.compile_friendly_flex_attention")
    @mock.patch(
        "torchtune.modules.attention_utils.nn.functional.scaled_dot_product_attention"
    )
    def test_flex_attention(self, mock_sdpa, mock_flex):
        # [b, n_h, s, h_d]
        q = torch.ones(2, 1, 3, 4)
        k = torch.ones(2, 1, 3, 4)
        v = torch.ones(2, 1, 3, 4)
        attn_mask = torch.ones(2, 3, 3)
        dropout_p = 0.0
        is_causal = False

        # Pretend that mask is actually a BlockMask
        with mock.patch(
            "torchtune.modules.attention_utils.isinstance", return_value=True
        ):
            _attention_call = _sdpa_or_flex_attention()
            _ = _attention_call(q, k, v, attn_mask, dropout_p, is_causal)
            mock_sdpa.assert_not_called()
            mock_flex.assert_called_with(q, k, v, block_mask=attn_mask)
        # If mask is not a BlockMask, then we should call SDPA
        _attention_call = _sdpa_or_flex_attention()
        _ = _attention_call(q, k, v, attn_mask, dropout_p, is_causal)
        mock_sdpa.assert_called_once()
        assert mock_flex.call_count == 1

    @mock.patch("torchtune.modules.attention_utils._SUPPORTS_FLEX_ATTENTION", False)
    @mock.patch(
        "torchtune.modules.attention_utils.nn.functional.scaled_dot_product_attention"
    )
    def test_sdpa_attention(self, mock_sdpa):
        # [b, n_h, s, h_d]
        q = torch.ones(2, 1, 3, 4)
        k = torch.ones(2, 1, 3, 4)
        v = torch.ones(2, 1, 3, 4)
        attn_mask = torch.ones(2, 3, 3)
        dropout_p = 0.0
        is_causal = False
        _attention_call = _sdpa_or_flex_attention()
        _ = _attention_call(q, k, v, attn_mask, dropout_p, is_causal)
        mock_sdpa.assert_called_once()
