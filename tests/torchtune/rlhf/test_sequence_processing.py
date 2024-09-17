# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtune import rlhf


class TestTruncateSequenceAtFirstStopToken:
    def test_truncate_sequences(self):
        stop_token_ids = torch.tensor([2, 869])
        fill_value = 0
        sequences = torch.tensor(
            [
                [869, 30, 869],
                [2, 30, 869],
                [869, 30, 2],
                [50, 30, 869],
                [13, 30, 2],
                [13, 30, 5],
                [13, 2, 20],
                [13, 2, 2],
                [2, 2, 2],
            ]
        )
        eos_mask, truncated_sequences = rlhf.truncate_sequence_at_first_stop_token(
            sequences, stop_token_ids, fill_value
        )

        expected_eos_mask = torch.tensor(
            [
                [False, True, True],
                [False, True, True],
                [False, True, True],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, True],
                [False, True, True],
            ]
        )

        expected_sequences = torch.tensor(
            [
                [869, fill_value, fill_value],
                [2, fill_value, fill_value],
                [869, fill_value, fill_value],
                [50, 30, 869],
                [13, 30, 2],
                [13, 30, 5],
                [13, 2, fill_value],
                [13, 2, fill_value],
                [2, fill_value, fill_value],
            ]
        )

        assert expected_eos_mask.eq(eos_mask).all()
        assert expected_sequences.eq(truncated_sequences).all()
