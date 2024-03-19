# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import (
    AlpacaInstructTemplate,
    GrammarErrorCorrectionTemplate,
    LLaMA2ChatTemplate,
    SummarizeTemplate,
)
from torchtune.data._templates import _Llama2ChatFormatConstants


class TestAlpacaInstructTemplate:
    samples = [
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": (
                "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables."
                "2. Exercise regularly to keep your body active and strong."
                "3. Get enough sleep and maintain a consistent sleep schedule."
            ),
        },
        {
            "instruction": "Evaluate this sentence for spelling and grammar mistakes",
            "input": "He finnished his meal and left the resturant",
            "output": "He finished his meal and left the restaurant.",
        },
    ]
    expected_prompts = [
        (
            "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\nGive three tips for staying healthy.\n\n"
            "### Response:\n"
        ),
        (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nEvaluate this sentence for spelling and grammar mistakes\n\n"
            "### Input:\nHe finnished his meal and left the resturant\n\n"
            "### Response:\n"
        ),
    ]

    template = AlpacaInstructTemplate()

    def test_format(self):
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            actual = self.template.format(sample)
            assert actual == expected_prompt

    def test_format_with_column_map(self):
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
            "Summarize this dialog:\n"
            "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)\n"
            "---\n"
            "Summary:\n"
        ),
        (
            "Summarize this dialog:\n"
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
        column_map = {"dialogue": "not_an_dialogue"}
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            modified_sample = sample.copy()
            modified_sample["not_an_dialogue"] = modified_sample["dialogue"]
            del modified_sample["dialogue"]

            actual = self.template.format(modified_sample, column_map=column_map)

            assert actual == expected_prompt


class TestLLaMA2ChatTemplate:
    samples = [
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "hi",
                },
                {
                    "from": "human",
                    "value": "mid",
                },
                {
                    "from": "gpt",
                    "value": "lo",
                },
            ]
        },
        {
            "conversations": [
                {
                    "from": "system",
                    "value": "one",
                },
                {
                    "from": "human",
                    "value": "two",
                },
                {
                    "from": "gpt",
                    "value": "three",
                },
            ]
        },
    ]
    expected_prompts = [
        f"{_Llama2ChatFormatConstants.B_INST} {_Llama2ChatFormatConstants.B_SYS}hi{_Llama2ChatFormatConstants.E_SYS}mid {_Llama2ChatFormatConstants.E_INST}",  # noqa: B950
        f"{_Llama2ChatFormatConstants.B_INST} {_Llama2ChatFormatConstants.B_SYS}one{_Llama2ChatFormatConstants.E_SYS}two {_Llama2ChatFormatConstants.E_INST}",  # noqa: B950
    ]
    template = LLaMA2ChatTemplate()

    def test_format(self):
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            actual = self.template.format(sample)
            assert actual == expected_prompt

    def test_format_with_column_map(self):
        column_map = {"conversations": "not_an_conversations"}
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            modified_sample = sample.copy()
            modified_sample["not_an_conversations"] = modified_sample["conversations"]
            del modified_sample["conversations"]

            actual = self.template.format(modified_sample, column_map=column_map)

            assert actual == expected_prompt
