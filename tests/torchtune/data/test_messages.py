# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest

from PIL import Image
from tests.test_utils import (
    assert_dialogue_equal,
    CHAT_SAMPLE,
    MESSAGE_SAMPLE,
    MESSAGE_SAMPLE_TRAIN_ON_INPUT,
)
from torchtune.data._messages import (
    ChosenRejectedToMessages,
    InputOutputToMessages,
    Message,
    OpenAIToMessages,
    ShareGPTToMessages,
    validate_messages,
)


class TestMessage:
    @pytest.fixture
    def text_message(self):
        return Message(role="user", content="hello world")

    @pytest.fixture
    def test_image(self):
        return Image.new(mode="RGB", size=(4, 4))

    @pytest.fixture
    def image_message(self, test_image):
        return Message(
            role="user",
            content=[
                {"type": "text", "content": "hello"},
                {"type": "image", "content": test_image},
                {"type": "text", "content": " world"},
            ],
        )

    def test_message_validation(self, text_message, test_image):
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
                content=[{"type": "image", "content": test_image}],
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

    def test_get_media(self, text_message, image_message, test_image):
        assert text_message.get_media() == []
        assert image_message.get_media() == [test_image]

    def test_text_content(self, text_message, image_message):
        assert text_message.text_content == "hello world"
        assert image_message.text_content == "hello world"

    def test_repr_text(self, text_message):
        expected_repr = "Message(role='user', content=['hello world'])"
        assert str(text_message) == expected_repr
        assert repr(text_message) == expected_repr

    def test_repr_image(self, image_message, test_image):
        img_repr = str(test_image)
        expected_repr = f"Message(role='user', content=['hello', {img_repr}, ' world'])"
        assert str(image_message) == expected_repr
        assert repr(image_message) == expected_repr


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
            Message(role="user", content="hello world", masked=True, eot=True),
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
            Message(role="user", content="hello world", masked=False, eot=True),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    def test_system_prompt(self, sample):
        transform = InputOutputToMessages(
            column_map={"input": "maybe_input", "output": "maybe_output"},
            new_system_prompt="you are a robot",
        )
        actual = transform(sample)
        expected = [
            Message(role="system", content="you are a robot", masked=True, eot=True),
            Message(role="user", content="hello world", masked=True, eot=True),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    def test_raise_value_error_when_input_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'input'"):
            InputOutputToMessages(
                column_map={"bananas": "maybe_input", "output": "maybe_output"},
            )

    def test_raise_value_error_when_output_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'output'"):
            InputOutputToMessages(
                column_map={"input": "maybe_input", "bananas": "maybe_output"},
            )


class TestChosenRejectedToMessages:
    @pytest.fixture
    def sample(self):
        return {
            "maybe_chosen": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hello world"},
            ],
            "maybe_rejected": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "bye world"},
            ],
        }

    def test_call(self, sample):
        transform = ChosenRejectedToMessages(
            column_map={
                "chosen": "maybe_chosen",
                "rejected": "maybe_rejected",
            },
        )
        actual = transform(sample)
        expected_chosen = [
            Message(role="user", content="hello world", masked=True, eot=True),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["chosen"], expected_chosen)

        expected_rejected = [
            Message(role="user", content="hello world", masked=True, eot=True),
            Message(role="assistant", content="bye world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["rejected"], expected_rejected)

    def test_call_train_on_input(self, sample):
        transform = ChosenRejectedToMessages(
            column_map={
                "chosen": "maybe_chosen",
                "rejected": "maybe_rejected",
            },
            train_on_input=True,
        )
        actual = transform(sample)
        expected_chosen = [
            Message(role="user", content="hello world", masked=False, eot=True),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["chosen"], expected_chosen)

        expected_rejected = [
            Message(role="user", content="hello world", masked=False, eot=True),
            Message(role="assistant", content="bye world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["rejected"], expected_rejected)

    def test_system_prompt(self, sample):
        transform = ChosenRejectedToMessages(
            column_map={
                "chosen": "maybe_chosen",
                "rejected": "maybe_rejected",
            },
            new_system_prompt="you are a robot",
        )
        actual = transform(sample)
        expected_chosen = [
            Message(role="system", content="you are a robot", masked=True, eot=True),
            Message(role="user", content="hello world", masked=True, eot=True),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["chosen"], expected_chosen)

        expected_rejected = [
            Message(role="system", content="you are a robot", masked=True, eot=True),
            Message(role="user", content="hello world", masked=True, eot=True),
            Message(role="assistant", content="bye world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["rejected"], expected_rejected)

    def test_raise_value_error_when_chosen_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'chosen'"):
            ChosenRejectedToMessages(
                column_map={"bananas": "maybe_chosen", "rejected": "maybe_rejected"},
            )

    def test_raise_value_error_when_rejected_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'rejected'"):
            ChosenRejectedToMessages(
                column_map={"chosen": "maybe_chosen", "bananas": "maybe_rejected"},
            )


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

    def test_call(self):
        transform = ShareGPTToMessages()
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], MESSAGE_SAMPLE)

    def test_call_train_on_input(self):
        transform = ShareGPTToMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"], MESSAGE_SAMPLE_TRAIN_ON_INPUT
        )

    def test_system_prompt(self):
        transform = ShareGPTToMessages(new_system_prompt="you are a robot")
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"],
            [
                Message(
                    role="system", content="you are a robot", masked=True, eot=True
                ),
            ]
            + MESSAGE_SAMPLE[1:],
        )

    def test_raise_value_error_when_conversations_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'conversations'"):
            ShareGPTToMessages(
                column_map={"bananas": "maybe_conversations"},
            )

    @mock.patch("torchtune.data._messages.load_image")
    def test_image_only_in_first_user_message(self, mock_load_image):
        mock_load_image.return_value = Image.new(mode="RGB", size=(4, 4))
        sample = {
            "conversations": [
                {"from": "human", "value": "<image>\nFirst message."},
                {"from": "gpt", "value": "First response."},
                {"from": "human", "value": "Second message."},
                {"from": "gpt", "value": "Second response."},
            ],
            "image": "test_image.jpg",
        }
        transform = ShareGPTToMessages(image_tag="<image>")
        messages = transform(sample)
        for idx, message in enumerate(messages["messages"]):
            if idx == 0:
                assert message.contains_media
            else:
                assert not message.contains_media


class TestOpenAIToMessages:
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

    image_samples = {
        "messages": [
            {
                "role": "system",
                "content": CHAT_SAMPLE["system"],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CHAT_SAMPLE["user"]},
                    {"type": "image_url", "image_url": {"url": "https://example.com"}},
                ],
            },
            {
                "role": "assistant",
                "content": CHAT_SAMPLE["assistant"],
            },
        ],
    }

    def test_call(self):
        transform = OpenAIToMessages()
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], MESSAGE_SAMPLE)

    def test_call_train_on_input(self):
        transform = OpenAIToMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"], MESSAGE_SAMPLE_TRAIN_ON_INPUT
        )

    def test_system_prompt(self):
        transform = OpenAIToMessages(new_system_prompt="you are a robot")
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"],
            [
                Message(
                    role="system", content="you are a robot", masked=True, eot=True
                ),
            ]
            + MESSAGE_SAMPLE[1:],
        )

    def test_raise_value_error_when_messages_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'messages'"):
            OpenAIToMessages(
                column_map={"bananas": "maybe_messages"},
            )

    @mock.patch("torchtune.data._messages.load_image")
    def test_convert_from_openai_content(self, mock_load_image):
        test_img = Image.new(mode="RGB", size=(4, 4))
        mock_load_image.return_value = test_img
        transform = OpenAIToMessages()
        converted_content = transform._convert_from_openai_content(
            self.image_samples["messages"][1]["content"]
        )
        assert converted_content == [
            {"type": "text", "content": CHAT_SAMPLE["user"]},
            {"type": "image", "content": test_img},
        ]
        mock_load_image.assert_called_once_with("https://example.com")

    @mock.patch("torchtune.data._messages.load_image")
    def test_call_image_messages(self, mock_load_image):
        test_img = Image.new(mode="RGB", size=(4, 4))
        mock_load_image.return_value = test_img
        transform = OpenAIToMessages()
        converted_messages = transform(self.image_samples)
        assert_dialogue_equal(
            converted_messages["messages"],
            [
                MESSAGE_SAMPLE[0],
                Message(
                    role="user",
                    content=[
                        {"type": "text", "content": CHAT_SAMPLE["user"]},
                        {"type": "image", "content": test_img},
                    ],
                ),
                MESSAGE_SAMPLE[2],
            ],
        )
        mock_load_image.assert_called_once_with("https://example.com")


def test_validate_messages():
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]

    # Test valid conversation with system
    validate_messages(messages)

    # Test valid conversation without system
    validate_messages(messages[1:])

    # Test system not first
    messages = [
        Message(role="user", content="hello"),
        Message(role="system", content="hello"),
        Message(role="assistant", content="world"),
    ]
    with pytest.raises(
        ValueError,
        match="System message at index 1 in messages, but system messages must come first",
    ):
        validate_messages(messages)

    # Test empty assistant message
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="world"),
        Message(role="assistant", content=""),
    ]
    validate_messages(messages)

    # Test single message
    messages = [
        Message(role="user", content="hello"),
    ]
    with pytest.raises(
        ValueError, match="Messages must be at least length 2, but got 1 messages"
    ):
        validate_messages(messages)

    # Test repeated user message
    messages = [
        Message(role="user", content="hello"),
        Message(role="user", content="world"),
        Message(role="assistant", content="world"),
    ]
    with pytest.raises(
        ValueError, match="Two consecutive user messages at index 1 and 0 in messages"
    ):
        validate_messages(messages)

    # Test assistant message comes first
    messages = [
        Message(role="assistant", content="hello"),
        Message(role="user", content="world"),
    ]
    with pytest.raises(
        ValueError,
        match="Assistant message before expected user message at index 0 in messages",
    ):
        validate_messages(messages)
