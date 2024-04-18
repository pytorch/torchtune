# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.modules.tokenizers._utils import _split_long_repetitions


class TestUtils:
    def test_split_long_repetitions(self):
        normal_str = "Here is a normal string"
        ten_spaces = "".join(10 * [" "])
        space_str = ten_spaces.join(
            ["Here", "is", "a", "string", "with", "long", "spaces"]
        )
        no_space_str = "".join(10 * ["ab"])

        actual_split = _split_long_repetitions(normal_str, 5)
        expected_split = ["Here is a norma", "l strin", "g"]
        for actual_substr, expected_substr in zip(actual_split, expected_split):
            assert actual_substr == expected_substr
        with pytest.raises(StopIteration):
            next(actual_split)

        actual_split = _split_long_repetitions(space_str, 9)
        expected_split = [
            "Here" + ten_spaces[:-1],
            " is" + ten_spaces[:-1],
            " a" + ten_spaces[:-1],
            " string" + ten_spaces[:-1],
            " with" + ten_spaces[:-1],
            " long" + ten_spaces[:-1],
            " spaces",
        ]
        for actual_substr, expected_substr in zip(actual_split, expected_split):
            assert actual_substr == expected_substr
        with pytest.raises(StopIteration):
            next(actual_split)

        actual_split = _split_long_repetitions(no_space_str, 4)
        expected_split = ["abab"] * 5
        for actual_substr, expected_substr in zip(actual_split, expected_split):
            assert actual_substr == expected_substr
        with pytest.raises(StopIteration):
            next(actual_split)
