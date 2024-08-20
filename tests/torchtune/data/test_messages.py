# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.test_utils import (
    assert_dialogue_equal,
    CHAT_SAMPLE,
    MESSAGE_SAMPLE,
    MESSAGE_SAMPLE_TRAIN_ON_INPUT,
)
from torchtune.data._messages import (
    ChosenRejectedToMessages,
    InputOutputToMessages,
    JSONToMessages,
    Message,
    ShareGPTToMessages,
)


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


class TestJSONToMessages:
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

    def test_call(self):
        transform = JSONToMessages()
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], MESSAGE_SAMPLE)

    def test_call_train_on_input(self):
        transform = JSONToMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"], MESSAGE_SAMPLE_TRAIN_ON_INPUT
        )

    def test_system_prompt(self):
        transform = JSONToMessages(new_system_prompt="you are a robot")
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
            JSONToMessages(
                column_map={"bananas": "maybe_messages"},
            )
