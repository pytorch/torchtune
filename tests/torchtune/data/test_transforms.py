# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import assert_dialogue_equal
from torchtune.data._transforms import (
    AlpacaInstructTemplate,
    ChatMLTemplate,
    GrammarErrorCorrectionTemplate,
    JsonToMessages,
    Llama2ChatTemplate,
    MistralChatTemplate,
    ShareGptToMessages,
    SummarizeTemplate,
)
from torchtune.data._types import Message

# Taken from Open-Orca/SlimOrca-Dedup on Hugging Face:
# https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup
CHAT_SAMPLE = {
    "system": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",  # noqa: B950
    "user": "Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? How about on an icy road? Well one father in Russia did just that, and recorded the entire thing. To her credit, the child seemed to be doing a great job. (0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\nSummary:",  # noqa: B950
    "assistant": "A father in Russia allowed his 8-year-old child to drive his car on an icy road and recorded the event. The child appeared to be handling the situation well, showcasing their driving skills despite the challenging conditions.",  # noqa: B950
}

CHAT_MESSAGES = {
    "messages": [
        Message(role="system", content=CHAT_SAMPLE["system"], masked=True),
        Message(role="user", content=CHAT_SAMPLE["user"], masked=True),
        Message(
            role="assistant",
            content=CHAT_SAMPLE["assistant"],
        ),
    ]
}

EXPECTED_MESSAGE_TRAIN_ON_INPUT = [
    Message(
        role="system",
        content=CHAT_SAMPLE["system"],
    ),
    Message(
        role="user",
        content=CHAT_SAMPLE["user"],
    ),
    Message(
        role="assistant",
        content=CHAT_SAMPLE["assistant"],
    ),
]

EXPECTED_MESSAGE = CHAT_MESSAGES["messages"]


class TestLlama2ChatTemplate:
    expected_dialogue = [
        Message(
            role="user",
            content="[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. "
            "Your goal is to complete the task as faithfully as you can. While performing "
            "the task think step-by-step and justify your steps.\n<</SYS>>\n\nPlease "
            "briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary: [/INST] ",
        ),
        Message(
            role="assistant",
            content="A father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.",
        ),
    ]

    def test_format(self):
        actual = Llama2ChatTemplate()(**CHAT_MESSAGES)
        assert_dialogue_equal(actual["messages"], self.expected_dialogue)


class TestMistralChatTemplate:
    expected_dialogue = [
        Message(
            role="user",
            content="[INST] Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old "
            "Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? "
            "How about on an icy road? Well one father in Russia did just that, and recorded "
            "the entire thing. To her credit, the child seemed to be doing a great job. "
            "(0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\n"
            "Summary: [/INST] ",
        ),
        Message(
            role="assistant",
            content="A father in Russia allowed his 8-year-old child to drive his car on an "
            "icy road and recorded the event. The child appeared to be handling the situation well, "
            "showcasing their driving skills despite the challenging conditions.",
        ),
    ]

    def test_format(self):
        no_system_sample = CHAT_MESSAGES[1:]
        actual = MistralChatTemplate()(**no_system_sample)
        assert_dialogue_equal(actual["messages"], self.expected_dialogue)

    def test_format_with_system_prompt_raises(self):
        with pytest.raises(
            ValueError, match="System prompts are not supported in MistralChatTemplate"
        ):
            _ = MistralChatTemplate()(**CHAT_MESSAGES)


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
        actual = ChatMLTemplate()(**CHAT_MESSAGES)
        assert_dialogue_equal(actual["messages"], self.expected_dialogue)


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
    expected_dialogues = [
        [
            Message(
                role="user",
                content="Below is an instruction that describes a task. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\nGive three tips for staying healthy.\n\n"
                "### Response:\n",
            ),
            Message(
                role="assistant",
                content="1.Eat a balanced diet and make sure to include plenty of fruits and vegetables."
                "2. Exercise regularly to keep your body active and strong."
                "3. Get enough sleep and maintain a consistent sleep schedule.",
            ),
        ],
        [
            Message(
                role="user",
                content="Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\nEvaluate this sentence for spelling and grammar mistakes\n\n"
                "### Input:\nHe finnished his meal and left the resturant\n\n"
                "### Response:\n",
            ),
            Message(
                role="assistant",
                content="He finished his meal and left the restaurant.",
            ),
        ],
    ]

    template = AlpacaInstructTemplate()

    def test_format(self):
        for sample, expected_dialogue in zip(self.samples, self.expected_dialogues):
            actual = self.template(**sample)
            assert_dialogue_equal(actual["messages"], expected_dialogue)


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
    expected_dialogues = [
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

    def test_format(self):
        for sample, expected_dialogue in zip(self.samples, self.expected_dialogues):
            actual = self.template(**sample)
            assert_dialogue_equal(actual["messages"], expected_dialogue)


class TestSummarizeTemplate:
    samples = [
        {
            "id": "13818513",
            "input": "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)",
            "output": "Amanda baked cookies and will bring Jerry some tomorrow.",
        },
        {
            "id": "13728867",
            "input": "Olivia: Who are you voting for in this election? Oliver: Liberals as always. Olivia: Me too!! Oliver: Great",  # noqa: B950
            "output": "Olivia and Olivier are voting for liberals in this election.",
        },
    ]
    expected_dialogues = [
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

    def test_format(self):
        for sample, expected_dialogue in zip(self.samples, self.expected_dialogues):
            actual = self.template(**sample)
            assert_dialogue_equal(actual["messages"], expected_dialogue)


class TestShareGptToMessages:
    samples = {
        "conversations": [
            {
                "from": "system",
                "value": CHAT_SAMPLE["system"],
            },
            {
                "from": "human",
                "value": CHAT_SAMPLE["user"],
            },
            {
                "from": "gpt",
                "value": CHAT_SAMPLE["assistant"],
            },
        ]
    }

    def test_conversion(self):
        transform = ShareGptToMessages()
        converted_messages = transform(**self.samples)
        for converted, expected in zip(converted_messages, EXPECTED_MESSAGE):
            assert converted == expected

    def test_conversion_train_on_input(self):
        transform = ShareGptToMessages(train_on_input=True)
        converted_messages = transform(**self.samples)
        for converted, expected in zip(
            converted_messages, EXPECTED_MESSAGE_TRAIN_ON_INPUT
        ):
            assert converted == expected


class TestJsonToMessages:
    samples = {
        "id": "DUMMY",
        "messages": [
            {
                "role": "system",
                "content": CHAT_SAMPLE["system"],
            },
            {
                "role": "user",
                "content": CHAT_SAMPLE["user"],
            },
            {
                "role": "assistant",
                "content": CHAT_SAMPLE["assistant"],
            },
        ],
    }

    def test_conversion(self):
        transform = JsonToMessages()
        converted_messages_2 = transform(**self.samples)
        for converted, expected in zip(converted_messages_2, EXPECTED_MESSAGE):
            assert converted == expected

    def test_conversion_train_on_input(self):
        transform = JsonToMessages(train_on_input=True)
        converted_messages_2 = transform(**self.samples)
        for converted, expected in zip(
            converted_messages_2, EXPECTED_MESSAGE_TRAIN_ON_INPUT
        ):
            assert converted == expected
