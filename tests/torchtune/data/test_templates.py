# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import (
    AlpacaInstructTemplate,
    GrammarErrorCorrectionTemplate,
    SummarizeTemplate,
)


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
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            modified_sample = sample.copy()
            modified_sample["not_an_instruction"], modified_sample["not_an_input"] = (
                modified_sample["instruction"],
                modified_sample["input"],
            )
            del modified_sample["instruction"], modified_sample["input"]

            actual = self.template.format(modified_sample, column_map=column_map)

            assert actual == expected_prompt


class TestGrammarErrorCorrectionTemplate:
    samples = [
        {
            "input": "Bitcoin is for $7,094 this morning, which CoinDesk says.",
            "output": "Bitcoin goes for $7,094 this morning, according to CoinDesk.",
        },
        {
            "input": "Much many brands and sellers still in the market.",
            "output": "Many brands and sellers still in the market.",
        },
    ]
    expected_prompts = [
        (
            "Correct this to standard English: Bitcoin is for $7,094 this morning, which CoinDesk says.\n"
            "---\n"
            "Corrected: "
        ),
        (
            "Correct this to standard English: Much many brands and sellers still in the market.\n"
            "---\n"
            "Corrected: "
        ),
    ]

    template = GrammarErrorCorrectionTemplate()

    def test_format(self):
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            column_map = {"sentence": "input"}
            actual = self.template.format(sample, column_map=column_map)
            assert actual == expected_prompt


class TestSummarizeTemplate:
    samples = [
        {
            "id": "13818513",
            "dialogue": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)",
            "summary": "Amanda baked cookies and will bring Jerry some tomorrow.",
        },
        {
            "id": "13728867",
            "dialogue": "Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great",  # noqa: B950
            "summary": "Olivia and Olivier are voting for liberals in this election.",
        },
    ]
    expected_prompts = [
        (
            "Summarize this dialogue:\n"
            "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)\n"
            "---\n"
            "Summary:\n"
        ),
        (
            "Summarize this dialogue:\n"
            "Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great\n"
            "---\n"
            "Summary:\n"
        ),
    ]

    template = SummarizeTemplate()

    def test_format(self):
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            actual = self.template.format(sample)
            assert actual == expected_prompt

    def test_format_with_column_map(self):
        column_map = {"dialogue": "not_a_dialogue"}
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            modified_sample = sample.copy()
            modified_sample["not_a_dialogue"] = modified_sample["dialogue"]
            del modified_sample["dialogue"]

            actual = self.template.format(modified_sample, column_map=column_map)

            assert actual == expected_prompt
