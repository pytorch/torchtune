# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.common import ASSETS
from torchtune.data._messages import Message
from torchtune.models.llama4 import Llama4Tokenizer


class TestLlama4Tokenizer:
    @pytest.fixture
    def tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return Llama4Tokenizer(
            path=str(ASSETS / "tiktoken_small_llama4.model"),
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

        # fmt: off
        expected_tokens = [200000, 200005, 477, 273, 200006, 10, 10, 73, 503, 655, 262, 376, 110, 46, 691, 721, 428, 270, 1122, 655, 262, 376, 110, 44, 270, 687, 334, 312, 522, 511, 115, 46, 200008] # noqa
        # fmt: on

        return message, expected_tokens

    @pytest.fixture
    def assistant_text_message(self, assistant_text):
        message = Message(
            role="assistant",
            content=assistant_text,
            masked=False,
            eot=True,
        )

        # fmt: off
        expected_tokens = [200005, 520, 511, 446, 200006, 10, 10, 65, 269, 277, 687, 334, 262, 376, 110, 351, 443, 32, 45, 334, 351, 1964, 46, 200008, 200001]  # noqa
        # fmt: on

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

        # fmt: off
        expected_tokens = [200005, 520, 511, 446, 200006, 10, 10, 525, 99, 534, 95, 115, 433, 40, 114, 338, 105, 477, 61, 49, 1642, 95, 1642, 48, 95, 1642, 48, 41, 200007]  # noqa
        # fmt: on

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

        # fmt: off
        expected_tokens = [200005, 1562, 121, 483, 279, 200006, 10, 10, 123, 34, 99, 960, 317, 34, 58, 323, 114, 982, 125, 200007]  # noqa
        # fmt: on

        return message, expected_tokens

    def test_token_ids(self, tokenizer):
        assert tokenizer.bos_id == 200000
        assert tokenizer.eos_id == 200001
        assert tokenizer.pad_id == 200018
        assert tokenizer.step_id == 200009
        assert tokenizer.start_header_id == 200005
        assert tokenizer.end_header_id == 200006
        assert tokenizer.eom_id == 200007
        assert tokenizer.eot_id == 200008
        assert tokenizer.image_id == 200090
        assert tokenizer.patch_id == 200092
        assert tokenizer.image_start == 200080
        assert tokenizer.image_end == 200081
        assert tokenizer.tile_x_separator == 200084
        assert tokenizer.tile_y_separator == 200085
        assert tokenizer.reasoning_start == 201142
        assert tokenizer.reasoning_end == 201143

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.base_vocab_size == 2000
        assert tokenizer.vocab_size == 202048

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
            _ = Llama4Tokenizer(
                path=str(ASSETS / "tiktoken_small_llama4.model"),
                # Same as LLAMA4_SPECIAL_TOKENS but one missing
                special_tokens={
                    "<|end_of_text|>": 200001,
                    "<|finetune_right_pad|>": 200018,
                    "<|step|>": 200009,
                    "<|header_start|>": 200005,
                    "<|header_end|>": 200006,
                    "<|eom|>": 200007,
                    "<|eot|>": 200008,
                    "<|python_start|>": 200016,
                    "<|python_end|>": 200017,
                    "<|patch|>": 200092,
                    "<|image|>": 200090,
                    "<|image_start|>": 200080,
                    "<|image_end|>": 200081,
                    "<|tile_x_separator|>": 200084,
                    "<|tile_y_separator|>": 200085,
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

    def test_get_tile_grid_tokens(
        self,
        tokenizer,
    ):
        """
        Corresponds to the following grid:
        <|image_start|>
        <|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|tile_y_separator|>
        <|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|tile_x_separator|><|patch|><|patch|><|patch|><|tile_y_separator|>
        <|image|><|patch|><|patch|><|patch|>
        <|image_end|>
        """
        # fmt: off
        expected_multi_tile = [
            200080,
            200092, 200092, 200092, 200084, 200092, 200092, 200092, 200084, 200092, 200092, 200092, 200085,
            200092, 200092, 200092, 200084, 200092, 200092, 200092, 200084, 200092, 200092, 200092, 200085,
            200090, 200092, 200092, 200092,
            200081,
        ]  # noqa
        # fmt: on
        actual = tokenizer._get_tile_grid_tokens(3, torch.tensor([2, 3]))
        assert actual == expected_multi_tile

        """
        Corresponds to the following grid:
        <|image_start|>
        <|image|><|patch|><|patch|><|patch|>
        <|image_end|>
        """
        # fmt: off
        expected_single_tile = [
            200080,
            200090, 200092, 200092, 200092,
            200081,
        ]  # noqa
        # fmt: on
        actual = tokenizer._get_tile_grid_tokens(3, torch.tensor([1, 1]))
        assert actual == expected_single_tile
