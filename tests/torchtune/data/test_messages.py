# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest

from PIL import Image
from tests.common import ASSETS
from tests.test_utils import (
    assert_dialogue_equal,
    CHAT_SAMPLE,
    MESSAGE_SAMPLE,
    MESSAGE_SAMPLE_TRAIN_ON_ASSISTANT,
    MESSAGE_SAMPLE_TRAIN_ON_INPUT,
    MESSAGE_SAMPLE_TRAIN_ON_LAST,
)
from torchtune.data._messages import (
    AlpacaToMessages,
    ChosenRejectedToMessages,
    InputOutputToMessages,
    mask_messages,
    Message,
    OpenAIToMessages,
    ShareGPTToMessages,
    validate_messages,
)

PYTORCH_RGB_IMAGE_AS_PIL = Image.open(ASSETS / "rgb_pytorch.png")


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

    @pytest.mark.parametrize(
        "input_image, expected_image",
        [
            ("rgb_pytorch.png", PYTORCH_RGB_IMAGE_AS_PIL),
            (ASSETS / "rgb_pytorch.png", PYTORCH_RGB_IMAGE_AS_PIL),
            (PYTORCH_RGB_IMAGE_AS_PIL, PYTORCH_RGB_IMAGE_AS_PIL),
        ],
    )
    def test_call_with_image(self, sample, input_image, expected_image):
        # Add the image to the sample
        sample["image"] = input_image

        # Create the transform
        transform = InputOutputToMessages(
            column_map={
                "input": "maybe_input",
                "output": "maybe_output",
                "image": "image",
            },
            # Need to test if the image_dir is properly joined w/ image
            image_dir=ASSETS if isinstance(input_image, str) else None,
        )
        actual = transform(sample)
        expected = [
            Message(
                role="user",
                content=[
                    {"type": "text", "content": "hello world"},
                    {"type": "image", "content": expected_image},
                ],
                masked=True,
                eot=True,
            ),
            Message(role="assistant", content="hello world", masked=False, eot=True),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    def test_call_with_image_fails_when_bad_image_inputs_are_passed(self, sample):
        # Construct a bad column_map without an 'image' key
        column_map = {
            "input": "maybe_input",
            "output": "maybe_output",
        }

        # Create a transform that expects an image column
        with pytest.raises(
            ValueError,
            match="Please specify an 'image' key in column_map",
        ):
            transform = InputOutputToMessages(
                column_map=column_map,
                image_dir=ASSETS,
            )

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

    @pytest.mark.parametrize(
        "masking_strategy, expected_masks",
        [
            ("train_on_all", [False, False]),
            ("train_on_assistant", [True, False]),
            ("train_on_last", [True, False]),
        ],
    )
    def test_call_masking_strategy(self, sample, masking_strategy, expected_masks):
        transform = InputOutputToMessages(
            column_map={"input": "maybe_input", "output": "maybe_output"},
            masking_strategy=masking_strategy,
        )
        actual = transform(sample)
        expected = [
            Message(
                role="user", content="hello world", masked=expected_masks[0], eot=True
            ),
            Message(
                role="assistant",
                content="hello world",
                masked=expected_masks[1],
                eot=True,
            ),
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

    @pytest.fixture
    def multi_turn_sample(self):
        return {
            "maybe_chosen": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "hello world"},
                {"role": "user", "content": "hello again world"},
                {"role": "assistant", "content": "hello again world"},
            ],
            "maybe_rejected": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "bye world"},
                {"role": "user", "content": "hello again world"},
                {"role": "assistant", "content": "bye again world"},
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

    @pytest.mark.parametrize(
        "masking_strategy, expected_masks",
        [
            ("train_on_all", [[False, False], [False, False]]),
            ("train_on_assistant", [[True, False], [True, False]]),
            ("train_on_last", [[True, False], [True, False]]),
        ],
    )
    def test_call_masking_strategy(self, sample, masking_strategy, expected_masks):
        transform = ChosenRejectedToMessages(
            column_map={
                "chosen": "maybe_chosen",
                "rejected": "maybe_rejected",
            },
            masking_strategy=masking_strategy,
        )
        actual = transform(sample)
        expected_chosen = [
            Message(
                role="user",
                content="hello world",
                masked=expected_masks[0][0],
                eot=True,
            ),
            Message(
                role="assistant",
                content="hello world",
                masked=expected_masks[0][1],
                eot=True,
            ),
        ]
        assert_dialogue_equal(actual["chosen"], expected_chosen)

        expected_rejected = [
            Message(
                role="user",
                content="hello world",
                masked=expected_masks[1][0],
                eot=True,
            ),
            Message(
                role="assistant",
                content="bye world",
                masked=expected_masks[1][1],
                eot=True,
            ),
        ]
        assert_dialogue_equal(actual["rejected"], expected_rejected)

    @pytest.mark.parametrize(
        "masking_strategy, expected_masks",
        [
            (
                "train_on_all",
                [[False, False, False, False], [False, False, False, False]],
            ),
            (
                "train_on_assistant",
                [[True, False, True, False], [True, False, True, False]],
            ),
            ("train_on_last", [[True, True, True, False], [True, True, True, False]]),
        ],
    )
    def test_call_masking_strategy_multi_turn(
        self, multi_turn_sample, masking_strategy, expected_masks
    ):
        transform = ChosenRejectedToMessages(
            column_map={
                "chosen": "maybe_chosen",
                "rejected": "maybe_rejected",
            },
            masking_strategy=masking_strategy,
        )
        actual = transform(multi_turn_sample)
        expected_chosen = [
            Message(
                role="user",
                content="hello world",
                masked=expected_masks[0][0],
                eot=True,
            ),
            Message(
                role="assistant",
                content="hello world",
                masked=expected_masks[0][1],
                eot=True,
            ),
            Message(
                role="user",
                content="hello again world",
                masked=expected_masks[0][2],
                eot=True,
            ),
            Message(
                role="assistant",
                content="hello again world",
                masked=expected_masks[0][3],
                eot=True,
            ),
        ]
        assert_dialogue_equal(actual["chosen"], expected_chosen)

        expected_rejected = [
            Message(
                role="user",
                content="hello world",
                masked=expected_masks[1][0],
                eot=True,
            ),
            Message(
                role="assistant",
                content="bye world",
                masked=expected_masks[1][1],
                eot=True,
            ),
            Message(
                role="user",
                content="hello again world",
                masked=expected_masks[1][2],
                eot=True,
            ),
            Message(
                role="assistant",
                content="bye again world",
                masked=expected_masks[1][3],
                eot=True,
            ),
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
    multi_turn_samples = {
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

    @pytest.mark.parametrize(
        "masking_strategy, expected_message",
        [
            ("train_on_all", MESSAGE_SAMPLE_TRAIN_ON_INPUT),
            ("train_on_assistant", MESSAGE_SAMPLE),
            ("train_on_last", MESSAGE_SAMPLE),
        ],
    )
    def test_call_masking_strategy(self, masking_strategy, expected_message):
        transform = ShareGPTToMessages(masking_strategy=masking_strategy)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], expected_message)

    @pytest.mark.parametrize(
        "masking_strategy, expected_message",
        [
            ("train_on_assistant", MESSAGE_SAMPLE_TRAIN_ON_ASSISTANT),
            ("train_on_last", MESSAGE_SAMPLE_TRAIN_ON_LAST),
        ],
    )
    def test_call_masking_strategy_multi_turn(self, masking_strategy, expected_message):
        transform = ShareGPTToMessages(masking_strategy=masking_strategy)
        converted_messages = transform(self.multi_turn_samples)
        assert_dialogue_equal(converted_messages["messages"], expected_message)

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

    multi_turn_samples = {
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
        transform = OpenAIToMessages()
        converted_messages = transform(self.samples)
        assert_dialogue_equal(converted_messages["messages"], MESSAGE_SAMPLE)

    def test_call_train_on_input(self):
        transform = OpenAIToMessages(train_on_input=True)
        converted_messages = transform(self.samples)
        assert_dialogue_equal(
            converted_messages["messages"], MESSAGE_SAMPLE_TRAIN_ON_INPUT
        )

    @mock.patch("torchtune.data._messages.load_image")
    @pytest.mark.parametrize(
        "masking_strategy, is_multimodal, expected_message",
        [
            ("train_on_all", False, MESSAGE_SAMPLE_TRAIN_ON_INPUT),
            ("train_on_assistant", False, MESSAGE_SAMPLE),
            ("train_on_last", False, MESSAGE_SAMPLE),
            ("train_on_all", True, None),
            ("train_on_assistant", True, None),
            ("train_on_last", True, None),
        ],
    )
    def test_call_masking_strategy(
        self, mock_load_image, masking_strategy, is_multimodal, expected_message
    ):
        test_img = Image.new(mode="RGB", size=(4, 4))
        mock_load_image.return_value = test_img
        transform = OpenAIToMessages(masking_strategy=masking_strategy)
        if is_multimodal:
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
                        masked=True,
                    ),
                    MESSAGE_SAMPLE[2],
                ],
            )
        else:
            converted_messages = transform(self.samples)
            assert_dialogue_equal(converted_messages["messages"], expected_message)

    @pytest.mark.parametrize(
        "masking_strategy, expected_message",
        [
            ("train_on_assistant", MESSAGE_SAMPLE_TRAIN_ON_ASSISTANT),
            ("train_on_last", MESSAGE_SAMPLE_TRAIN_ON_LAST),
        ],
    )
    def test_call_masking_strategy_multi_turn(self, masking_strategy, expected_message):
        transform = OpenAIToMessages(masking_strategy=masking_strategy)
        converted_messages = transform(self.multi_turn_samples)
        assert_dialogue_equal(converted_messages["messages"], expected_message)

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
                    masked=True,
                ),
                MESSAGE_SAMPLE[2],
            ],
        )
        mock_load_image.assert_called_once_with("https://example.com")

    def test_call_tool_messages(self):
        tool_samples = {
            "messages": [
                {
                    "role": "system",
                    "content": "Available functions: weather(city: str), search(query: str)",
                },
                {"role": "user", "content": "What's the weather in Istanbul?"},
                {"role": "assistant", "content": "weather(city='Istanbul')"},
                {"role": "tool", "content": "{'temperature': 25}"},
                {
                    "role": "assistant",
                    "content": "The temperature in Istanbul is 25°C.",
                },
            ]
        }
        transform = OpenAIToMessages()
        converted_messages = transform(tool_samples)
        assert_dialogue_equal(
            converted_messages["messages"],
            [
                Message(
                    role="system",
                    content="Available functions: weather(city: str), search(query: str)",
                ),
                Message(role="user", content="What's the weather in Istanbul?"),
                Message(role="assistant", content="weather(city='Istanbul')"),
                Message(
                    role="tool", content="{'temperature': 25}", eot=False, masked=True
                ),
                Message(
                    role="assistant", content="The temperature in Istanbul is 25°C."
                ),
            ],
        )


class TestAlpacaToMessages:
    template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n"
        ),
    }

    @pytest.fixture
    def sample(self):
        return {
            "maybe_instruction": "hello world",
            "maybe_input": "this is some input",
            "maybe_output": "this is some output",
        }

    @pytest.fixture
    def sample_with_no_input(self):
        return {
            "maybe_instruction": "hello world",
            "maybe_output": "this is some output",
        }

    def test_call(self, sample, sample_with_no_input):
        # input in column_map and in sample
        transform = AlpacaToMessages(
            column_map={
                "instruction": "maybe_instruction",
                "input": "maybe_input",
                "output": "maybe_output",
            }
        )
        actual = transform(sample)
        expected = [
            Message(
                role="user",
                content=self.template["prompt_input"].format(
                    instruction="hello world", input="this is some input"
                ),
                masked=False,
                eot=True,
            ),
            Message(
                role="assistant", content="this is some output", masked=False, eot=True
            ),
        ]
        assert_dialogue_equal(actual["messages"], expected)

        # input not in column_map and not in sample
        transform = AlpacaToMessages(
            column_map={"instruction": "maybe_instruction", "output": "maybe_output"}
        )
        actual = transform(sample_with_no_input)
        expected = [
            Message(
                role="user",
                content=self.template["prompt_no_input"].format(
                    instruction="hello world"
                ),
                masked=False,
                eot=True,
            ),
            Message(
                role="assistant", content="this is some output", masked=False, eot=True
            ),
        ]
        assert_dialogue_equal(actual["messages"], expected)

        # input not in column_map but in sample
        transform = AlpacaToMessages(
            column_map={"instruction": "maybe_instruction", "output": "maybe_output"}
        )
        actual = transform(sample)
        expected = [
            Message(
                role="user",
                content=self.template["prompt_no_input"].format(
                    instruction="hello world"
                ),
                masked=False,
                eot=True,
            ),
            Message(
                role="assistant", content="this is some output", masked=False, eot=True
            ),
        ]
        assert_dialogue_equal(actual["messages"], expected)

        # input in column_map but not in sample
        transform = AlpacaToMessages(
            column_map={
                "instruction": "maybe_instruction",
                "input": "maybe_input",
                "output": "maybe_output",
            }
        )
        actual = transform(sample_with_no_input)
        expected = [
            Message(
                role="user",
                content=self.template["prompt_no_input"].format(
                    instruction="hello world"
                ),
                masked=False,
                eot=True,
            ),
            Message(
                role="assistant", content="this is some output", masked=False, eot=True
            ),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    def test_call_train_on_input(self, sample_with_no_input):
        transform = AlpacaToMessages(
            column_map={
                "instruction": "maybe_instruction",
                "input": "maybe_input",
                "output": "maybe_output",
            },
            train_on_input=True,
        )
        actual = transform(sample_with_no_input)
        expected = [
            Message(
                role="user",
                content=self.template["prompt_no_input"].format(
                    instruction="hello world",
                ),
                masked=False,
                eot=True,
            ),
            Message(
                role="assistant", content="this is some output", masked=False, eot=True
            ),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    @pytest.mark.parametrize(
        "masking_strategy, expected_masks",
        [
            ("train_on_all", [False, False]),
            ("train_on_assistant", [True, False]),
            ("train_on_last", [True, False]),
        ],
    )
    def test_call_masking_strategy(self, sample, masking_strategy, expected_masks):
        transform = AlpacaToMessages(
            column_map={
                "instruction": "maybe_instruction",
                "input": "maybe_input",
                "output": "maybe_output",
            }
        )
        actual = transform(sample)
        expected = [
            Message(
                role="user",
                content=self.template["prompt_input"].format(
                    instruction="hello world", input="this is some input"
                ),
                masked=expected_masks[0],
                eot=True,
            ),
            Message(
                role="assistant",
                content="this is some output",
                masked=expected_masks[1],
                eot=True,
            ),
        ]
        assert_dialogue_equal(actual["messages"], expected)

    def test_raise_value_error_when_instruction_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'instruction'"):
            AlpacaToMessages(
                column_map={"bananas": "maybe_instruction", "output": "maybe_output"},
            )

    def test_raise_value_error_when_output_not_in_column_map(self):
        with pytest.raises(ValueError, match="Expected a key of 'output'"):
            AlpacaToMessages(
                column_map={
                    "instruction": "maybe_instruction",
                    "bananas": "maybe_output",
                },
            )


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

    # Test valid conversation with tool
    messages = [
        Message(
            role="system",
            content="Available functions: weather(city: str), search(query: str)",
        ),
        Message(role="user", content="What is the weather in Istanbul?"),
        Message(role="assistant", content="weather(city='Istanbul')", ipython=True),
        Message(role="tool", content="{'temperature': 25}"),
        Message(role="assistant", content="The weather in Istanbul is 25C"),
    ]
    validate_messages(messages)

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
        match="Assistant message before expected user, tool or ipython message at index 0 in messages",
    ):
        validate_messages(messages)

    # # Test tool message before ipython message
    messages = [
        Message(role="user", content="get weather for istanbul"),
        Message(role="assistant", content="get_weather(city='Istanbul')"),
        Message(role="ipython", content="{'temperature': 25}"),
    ]
    with pytest.raises(
        ValueError,
        match="Tool or ipython message at index 2 must follow an ipython message",
    ):
        validate_messages(messages)


@pytest.mark.parametrize(
    "masking_strategy, messages_count, expected_masks",
    [
        ("train_on_all", 3, [True, False, False]),
        ("train_on_assistant", 3, [True, True, False]),
        ("train_on_last", 3, [True, True, False]),
        ("train_on_all", 5, [True, False, False, False, False]),
        ("train_on_assistant", 5, [True, True, False, True, False]),
        ("train_on_last", 5, [True, True, True, True, False]),
        ("some_invalid_strategy", 3, None),
    ],
)
def test_mask_messages(masking_strategy, messages_count, expected_masks):
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]

    input_messages = messages[:messages_count]
    if masking_strategy == "some_invalid_strategy":
        with pytest.raises(
            ValueError,
            match="'some_invalid_strategy' is not a valid MaskingStrategy",
        ):
            mask_messages(input_messages, masking_strategy=masking_strategy)
    else:
        mask_messages(input_messages, masking_strategy=masking_strategy)
        expected_messages = messages[:messages_count]
        for index in range(messages_count):
            expected_messages[index].masked = expected_masks[index]
        assert_dialogue_equal(input_messages, expected_messages)
