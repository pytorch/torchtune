# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.test_utils import assert_dialogue_equal, MESSAGE_SAMPLE
from torchtune.data import (
    ChatMLTemplate,
    GrammarErrorCorrectionTemplate,
    Message,
    SummarizeTemplate,
)


class TestChatMLTemplate:
    expected_dialogue = [
        Message(
            role="system",
            content="<|im_start|>system\nYou are an AI assistant. User will you give you a task. "
            "Your goal is to complete the task as faithfully as you can. While performing "
            "the task think step-by-step and justify your steps.<|im_end|>\n",
        ),
        Message(
            role="user",
            content="<|im_start|>user\nPlease "
            "briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary:<|im_end|>\n",
        ),
        Message(
            role="assistant",
            content="<|im_start|>assistant\nA father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.<|im_end|>",
        ),
    ]

    def test_format(self):
        actual = ChatMLTemplate()(MESSAGE_SAMPLE)
        assert_dialogue_equal(actual, self.expected_dialogue)


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
