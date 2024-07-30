# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.test_utils import assert_dialogue_equal
from torchtune.data import GrammarErrorCorrectionTemplate, Message, SummarizeTemplate


class TestGrammarErrorCorrectionTemplate:
    samples = [
        {
            "messages": [
                Message(
                    role="user",
                    content="Bitcoin is for $7,094 this morning, which CoinDesk says.",
                ),
                Message(
                    role="assistant",
                    content="Bitcoin goes for $7,094 this morning, according to CoinDesk.",
                ),
            ]
        },
        {
            "messages": [
                Message(
                    role="user",
                    content="Much many brands and sellers still in the market.",
                ),
                Message(
                    role="assistant",
                    content="Many brands and sellers still in the market.",
                ),
            ],
        },
    ]
    expected_prompts = [
        [
            Message(
                role="user",
                content="Correct this to standard English: Bitcoin is for $7,094 this morning, which CoinDesk says.\n"
                "---\n"
                "Corrected: ",
            ),
            Message(
                role="assistant",
                content="Bitcoin goes for $7,094 this morning, according to CoinDesk.",
            ),
        ],
        [
            Message(
                role="user",
                content="Correct this to standard English: Much many brands and sellers still in the market.\n"
                "---\n"
                "Corrected: ",
            ),
            Message(
                role="assistant",
                content="Many brands and sellers still in the market.",
            ),
        ],
    ]

    template = GrammarErrorCorrectionTemplate()

    def test_call(self):
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            actual = self.template(sample["messages"])
            assert_dialogue_equal(actual, expected_prompt)


class TestSummarizeTemplate:
    samples = [
        {
            "messages": [
                Message(
                    role="user",
                    content="Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)",
                ),
                Message(
                    role="assistant",
                    content="Amanda baked cookies and will bring Jerry some tomorrow.",
                ),
            ],
        },
        {
            "messages": [
                Message(
                    role="user",
                    content="Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great",  # noqa: B950
                ),
                Message(
                    role="assistant",
                    content="Olivia and Olivier are voting for liberals in this election.",
                ),
            ],
        },
    ]
    expected_prompts = [
        [
            Message(
                role="user",
                content="Summarize this dialogue:\n"
                "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)\n"
                "---\n"
                "Summary:\n",
            ),
            Message(
                role="assistant",
                content="Amanda baked cookies and will bring Jerry some tomorrow.",
            ),
        ],
        [
            Message(
                role="user",
                content="Summarize this dialogue:\n"
                "Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great\n"
                "---\n"
                "Summary:\n",
            ),
            Message(
                role="assistant",
                content="Olivia and Olivier are voting for liberals in this election.",
            ),
        ],
    ]

    template = SummarizeTemplate()

    def test_call(self):
        for sample, expected_prompt in zip(self.samples, self.expected_prompts):
            actual = self.template(sample["messages"])
            assert_dialogue_equal(actual, expected_prompt)
