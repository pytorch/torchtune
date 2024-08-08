# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import AlpacaInstructTemplate


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
