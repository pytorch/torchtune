# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import truncate


def test_truncate():
    tokens = [1, 2, 3, 4, -1]

    # Test no truncation
    truncated_tokens = truncate(
        tokens=tokens,
        max_seq_len=5,
        eos_id=-1,
    )
    assert truncated_tokens == tokens

    masks = [True, True, False, True, False]
    # Test truncated mask
    truncated_masks = truncate(tokens=masks, max_seq_len=4, eos_id=False)
    assert truncated_masks == [True, True, False, False]
