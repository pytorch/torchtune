# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.common import ASSETS
from torchtune.data._messages import Message
from torchtune.models.llama3 import llama3_tokenizer, Llama3Tokenizer


class TestLlama3Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return llama3_tokenizer(
            path=str(ASSETS / "tiktoken_small.model"),
            max_seq_len=2048,
        )

    @pytest.fixture
    def user_text_a(self):
        return "I can see the sun. "

    @pytest.fixture
    def user_text_b(self):
        return "But even if I cannot see the sun, I know that it exists."

    @pytest.fixture
    def assistant_text(self):
        return "And to know that the sun is there - that is living."

    @pytest.fixture
    def user_text_message(self, user_text_a, user_text_b):
        message = Message(
            role="user",
            content=user_text_a + user_text_b,
            masked=True,
            eot=True,
        )
        expected_tokens = [
            128000,
            128006,
            477,
            273,
            128007,
            10,
            10,
            73,
            503,
            654,
            262,
            376,
            110,
            46,
            690,
            720,
            428,
            270,
            1119,
            654,
            262,
            376,
            110,
            44,
            270,
            686,
            334,
            312,
            522,
            511,
            115,
            46,
            128009,
        ]
        return message, expected_tokens

    @pytest.fixture
    def assistant_text_message(self, assistant_text):
        message = Message(
            role="assistant",
            content=assistant_text,
            masked=False,
            eot=True,
        )
        expected_tokens = [
            128006,
            520,
            511,
            446,
            128007,
            10,
            10,
            65,
            269,
            277,
            686,
            334,
            262,
            376,
            110,
            351,
            443,
            32,
            45,
            334,
            351,
            1955,
            46,
            128009,
            128001,
        ]
        return message, expected_tokens

    @pytest.fixture
    def user_image_text_message(self, user_text_a, user_text_b):
        message = Message(
            role="user",
            content=[
                {"type": "image"},
                {"type": "text", "content": user_text_a + user_text_b},
            ],
            masked=True,
            eot=True,
        )
        expected_tokens = [
            128000,
            128006,
            477,
            273,
            128007,
            10,
            10,
            128256,
            73,
            503,
            654,
            262,
            376,
            110,
            46,
            690,
            720,
            428,
            270,
            1119,
            654,
            262,
            376,
            110,
            44,
            270,
            686,
            334,
            312,
            522,
            511,
            115,
            46,
            128009,
        ]
        return message, expected_tokens

    @pytest.fixture
    def user_interleaved_image_text_message(self, user_text_a, user_text_b):
        message = Message(
            role="user",
            content=[
                {"type": "image"},
                {"type": "text", "content": user_text_a},
                {"type": "image"},
                {"type": "text", "content": user_text_b},
            ],
            masked=True,
            eot=True,
        )
        expected_tokens = [
            128000,
            128006,
            477,
            273,
            128007,
            10,
            10,
            128256,
            73,
            503,
            654,
            262,
            376,
            110,
            46,
            128256,
            1542,
            720,
            428,
            270,
            1119,
            654,
            262,
            376,
            110,
            44,
            270,
            686,
            334,
            312,
            522,
            511,
            115,
            46,
            128009,
        ]
        return message, expected_tokens

    @pytest.fixture
    def assistant_tool_message(self):
        message = Message(
            role="assistant",
            content=[
                {"type": "text", "content": "locate_sun(radius=100_000_000)"},
            ],
            masked=False,
            ipython=True,
            eot=False,
        )
        expected_tokens = [
            128006,
            520,
            511,
            446,
            128007,
            10,
            10,
            128010,
            525,
            99,
            534,
            95,
            115,
            433,
            40,
            114,
            338,
            105,
            477,
            61,
            49,
            1635,
            95,
            1635,
            48,
            95,
            1635,
            48,
            41,
            128008,
        ]
        return message, expected_tokens

    @pytest.fixture
    def ipython_message(self):
        message = Message(
            role="ipython",
            content=[
                {"type": "text", "content": '{"content": True}'},
            ],
            masked=True,
            eot=False,
        )
        expected_tokens = [
            128006,
            1558,
            121,
            483,
            279,
            128007,
            10,
            10,
            123,
            34,
            99,
            957,
            317,
            34,
            58,
            323,
            114,
            979,
            125,
            128008,
        ]
        return message, expected_tokens

    def test_token_ids(self, tokenizer):
        assert tokenizer.bos_id == 128000
        assert tokenizer.eos_id == 128001
        assert tokenizer.pad_id == 128004
        assert tokenizer.step_id == 128005
        assert tokenizer.start_header_id == 128006
        assert tokenizer.end_header_id == 128007
        assert tokenizer.eom_id == 128008
        assert tokenizer.eot_id == 128009
        assert tokenizer.python_tag == 128010
        assert tokenizer.image_id == 128256

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.base_vocab_size == 2000
        assert tokenizer.vocab_size == 128257

    def test_tokenize_text_messages(
        self, tokenizer, user_text_message, assistant_text_message
    ):
        text_messages = [user_text_message[0], assistant_text_message[0]]
        expected_tokens = user_text_message[1] + assistant_text_message[1]
        expected_mask = (
            [True] * len(user_text_message[1])
            + [False] * (len(assistant_text_message[1]) - 1)
            + [True]
        )
        tokens, mask = tokenizer.tokenize_messages(text_messages)
        assert tokens == expected_tokens
        assert mask == expected_mask

    def test_tokenize_message_drop_eot_and_eos(
        self, tokenizer, user_text_message, assistant_text_message
    ):
        """
        Test that the tokenizer will not add an EOS token or EOT token if user requests it.
        This is the most common case for inference.
        """
        text_messages = [user_text_message[0], assistant_text_message[0]]
        # Chop the end of the assistant message to remove the EOS token *and* EOT token
        expected_tokens = user_text_message[1] + assistant_text_message[1][:-2]
        # No need to mask out the EOS token *or* EOT token at the end since they are not there
        expected_mask = [True] * len(user_text_message[1]) + [False] * (
            len(assistant_text_message[1]) - 2
        )
        tokens, mask = tokenizer.tokenize_messages(text_messages, add_end_tokens=False)
        assert tokens == expected_tokens
        assert mask == expected_mask

    def test_tokenize_image_and_text_messages(
        self, tokenizer, user_image_text_message, assistant_text_message
    ):
        image_and_text_messages = [
            user_image_text_message[0],
            assistant_text_message[0],
        ]
        expected_tokens = user_image_text_message[1] + assistant_text_message[1]
        expected_mask = (
            [True] * len(user_image_text_message[1])
            + [False] * (len(assistant_text_message[1]) - 1)
            + [True]
        )
        tokens, mask = tokenizer.tokenize_messages(image_and_text_messages)
        assert tokens == expected_tokens
        assert mask == expected_mask

    def test_tokenize_interleaved_image_and_text_messages(
        self,
        tokenizer,
        user_interleaved_image_text_message,
        assistant_text_message,
    ):
        interleaved_image_and_text_messages = [
            user_interleaved_image_text_message[0],
            assistant_text_message[0],
        ]
        expected_tokens = (
            user_interleaved_image_text_message[1] + assistant_text_message[1]
        )
        expected_mask = (
            [True] * len(user_interleaved_image_text_message[1])
            + [False] * (len(assistant_text_message[1]) - 1)
            + [True]
        )
        tokens, mask = tokenizer.tokenize_messages(interleaved_image_and_text_messages)
        assert tokens == expected_tokens
        assert mask == expected_mask

    def test_tokenize_tool_call_messages(
        self,
        tokenizer,
        user_text_message,
        assistant_tool_message,
        ipython_message,
        assistant_text_message,
    ):
        tool_call_messages = [
            user_text_message[0],
            assistant_tool_message[0],
            ipython_message[0],
            assistant_text_message[0],
        ]
        expected_tokens = (
            user_text_message[1]
            + assistant_tool_message[1]
            + ipython_message[1]
            + assistant_text_message[1]
        )
        expected_mask = (
            [True] * len(user_text_message[1])
            + [False] * len(assistant_tool_message[1])
            + [True] * len(ipython_message[1])
            + [False] * (len(assistant_text_message[1]) - 1)
            + [True]
        )
        tokens, mask = tokenizer.tokenize_messages(tool_call_messages)
        assert tokens == expected_tokens
        assert mask == expected_mask

    def test_validate_special_tokens(self):
        with pytest.raises(
            ValueError, match="<|begin_of_text|> missing from special_tokens"
        ):
            _ = Llama3Tokenizer(
                path=str(ASSETS / "tiktoken_small.model"),
                # Same as LLAMA3_SPECIAL_TOKENS but one missing
                special_tokens={
                    "<|end_of_text|>": 128001,
                    "<|start_header_id|>": 128006,
                    "<|end_header_id|>": 128007,
                    "<|eot_id|>": 128009,
                    "<|eom_id|>": 128008,
                    "<|python_tag|>": 128255,
                },
            )

    def test_skip_special_tokens(
        self,
        tokenizer,
        user_text_message,
        assistant_text_message,
        user_text_a,
        user_text_b,
        assistant_text,
    ):
        # This should satisfy text = decode(encode(text))
        tokens = user_text_message[1] + assistant_text_message[1]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        assert text == user_text_a + user_text_b + assistant_text
