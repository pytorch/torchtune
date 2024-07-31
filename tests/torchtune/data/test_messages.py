# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import assert_dialogue_equal
from torchtune.data._messages import (
    JsonToMessages,
    Message,
    ShareGPTToMessages,
    InputOutputToMessages,
)

# Taken from Open-Orca/SlimOrca-Dedup on Hugging Face:
# https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup
CHAT_SAMPLE = {
    "system": "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.",  # noqa: B950
    "user": "Please briefly summarize this news article:\n\nAOL.com Video - Father Lets 8-Year-Old Drive On Icy Road\n\nDescription:Would you let your 8-year-old drive your car? How about on an icy road? Well one father in Russia did just that, and recorded the entire thing. To her credit, the child seemed to be doing a great job. (0:44)\n\nTags: 8-year-old driver , caught on camera , child driver , pix11\n\nSummary:",  # noqa: B950
    "assistant": "A father in Russia allowed his 8-year-old child to drive his car on an icy road and recorded the event. The child appeared to be handling the situation well, showcasing their driving skills despite the challenging conditions.",  # noqa: B950
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

EXPECTED_MESSAGE = [
    Message(role="system", content=CHAT_SAMPLE["system"], masked=True),
    Message(role="user", content=CHAT_SAMPLE["user"], masked=True),
    Message(
        role="assistant",
        content=CHAT_SAMPLE["assistant"],
    ),
]


class TestMessage:
    @pytest.fixture
    def text_message(self):
        return Message(role="user", content="hello world")

    @pytest.fixture
    def image_message(self):
        return Message(
            role="user",
            content=[
                {"type": "text", "content": "hello"},
                {"type": "image"},
                {"type": "text", "content": " world"},
            ],
        )

    def test_message_validation(self, text_message):
        message = text_message
        assert message.role == "user"
        assert message.content == [{"type": "text", "content": "hello world"}]

        with pytest.raises(
            ValueError,
            match="Only assistant messages can be tool calls. Found role user in message: hello world",
        ):
            message = Message(role="user", content="hello world", ipython=True)

        with pytest.raises(
            ValueError,
            match="Media tokens in tool calls are not supported. Both are set in message: ",
        ):
            message = Message(
                role="user",
                content=[{"type": "image"}],
                ipython=True,
            )

    def test_from_dict(self):
        message = Message.from_dict({"role": "user", "content": "hello world"})
        assert message.role == "user"
        assert message.content[0]["content"] == "hello world"
        assert not message.masked
        assert not message.ipython
        assert message.eot

    def test_contains_media(self, text_message, image_message):
        assert not text_message.contains_media
        assert image_message.contains_media

    def test_text_content(self, text_message, image_message):
        assert text_message.text_content == "hello world"
        assert image_message.text_content == "hello world"


class TestInputOutputToMessages:
    @pytest.fixture
    def sample(self):
        return {
            "maybe_input": "hello world",
            "maybe_output": "hello world",
        }

    def test_call(self, sample):
        transform = InputOutputToMessages(
            column_map={"input": "maybe_input", "output": "maybe_output"}
        )
        actual = transform(sample)
        expected = [
            Message(role="user", content="hello world", masked=True, eot=False),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    def test_call_train_on_input(self, sample):
        transform = InputOutputToMessages(
            column_map={"input": "maybe_input", "output": "maybe_output"},
            train_on_input=True,
        )
        actual = transform(sample)
        expected = [
            Message(role="user", content="hello world", masked=False, eot=False),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["messages"], expected)


class TestShareGPTToMessages:
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
        transform = ShareGPTToMessages()
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], EXPECTED_MESSAGE)

    def test_conversion_train_on_input(self):
        transform = ShareGPTToMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"], EXPECTED_MESSAGE_TRAIN_ON_INPUT
        )


class TestJsonToMessages:
    samples = {
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
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], EXPECTED_MESSAGE)

    def test_conversion_train_on_input(self):
        transform = JsonToMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"], EXPECTED_MESSAGE_TRAIN_ON_INPUT
        )
