# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data.templates import AlpacaInstructTemplate


class TestAlpacaInstructTemplate:
    def test_format(self):
        template = AlpacaInstructTemplate()
        sample = {
            "instruction": "A man, a plan, a canal, Panama!",
            "input": "The fox jumped over the lazy dog.",
        }
        # Test with instruction and input
        actual = template.format(sample)
        expected = template.system["prompt_input"].format(**sample)
        assert actual == expected
        # Test with instruction only
        sample = {
            "instruction": "A man, a plan, a canal, Panama!",
        }
        actual = template.format(sample)
        expected = template.system["prompt_no_input"].format(**sample)
        assert actual == expected

    def test_format_with_column_map(self):
        template = AlpacaInstructTemplate()
        sample = {
            "not_an_instruction": "A man, a plan, a canal, Panama!",
            "not_an_input": "The fox jumped over the lazy dog.",
        }
        column_map = {"instruction": "not_an_instruction", "input": "not_an_input"}
        # Test with instruction and input
        actual = template.format(sample, column_map=column_map)
        expected = template.system["prompt_input"].format(
            instruction=sample["not_an_instruction"], input=sample["not_an_input"]
        )
        assert actual == expected
