# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
from tests.test_utils import DummyTokenizer
from torchtune.data import Message
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX

from torchtune.datasets import ChatDataset


class DummyTemplate:
    def __init__(self):
        self.template = {
            "system": "System:\n{system}\nUser:\n{user}\nAssistant:\n",
            "no_system": "User:\n{user}\nAssistant:\n",
        }

    def format(self, sample, column_map=None):
        if "system" in sample:
            return self.template["system"].format(**sample)
        else:
            return self.template["no_system"].format(**sample)


def _are_messages_equal(messages_a, messages_b):
    for ma, mb in zip(messages_a, messages_b):
        if ma.role != mb.role:
            return False
        if ma.content != mb.content:
            return False
    return True


class TestChatDataset:
    @pytest.fixture
    def template(self):
        return DummyTemplate()

    @pytest.fixture
    def dialogue(self):
        return [
            {
                "dialogue": [
                    Message(role="system", content="You are an AI assistant."),
                    Message(role="user", content="What is the meaning of life?"),
                    Message(role="assistant", content="The meaning of life is 42."),
                    Message(role="user", content="That's ridiculous."),
                    Message(role="assistant", content="I agree."),
                ],
            },
        ]

    @mock.patch("torchtune.datasets._chat.load_dataset")
    def test_get_turns(self, mock_load_dataset, template, dialogue):
        ds = ChatDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            convert_to_dialogue=lambda x: x,
            template=template,
            max_seq_len=100,
            train_on_input=False,
        )

        # Test a normal multiturn dialogue
        prompts, responses = zip(
            *[(p, l) for p, l in ds._get_turns(dialogue[0]["dialogue"])]
        )
        assert prompts[0] == {
            "system": "You are an AI assistant.",
            "user": "What is the meaning of life?",
        }
        assert responses[0] == "The meaning of life is 42."
        assert prompts[1] == {"user": "That's ridiculous."}
        assert responses[1] == "I agree."

        # Test without system prompt
        prompts, responses = zip(
            *[(p, l) for p, l in ds._get_turns(dialogue[0]["dialogue"][1:])]
        )
        assert prompts[0] == {"user": "What is the meaning of life?"}
        assert responses[0] == "The meaning of life is 42."
        assert prompts[1] == {"user": "That's ridiculous."}
        assert responses[1] == "I agree."

        # Test a missing user message
        with pytest.raises(
            ValueError, match="Missing a user message before assistant message"
        ):
            for _ in ds._get_turns(
                [dialogue[0]["dialogue"][0]] + dialogue[0]["dialogue"][2:]
            ):
                pass

        # Test a missing user message and no system message
        with pytest.raises(
            ValueError, match="Missing a user message before assistant message"
        ):
            for _ in ds._get_turns(dialogue[0]["dialogue"][2:]):
                pass

        # Test repeated messages
        with pytest.raises(ValueError, match="Duplicate"):
            for _ in ds._get_turns(
                dialogue[0]["dialogue"][:2] + dialogue[0]["dialogue"][3:]
            ):
                pass
        with pytest.raises(ValueError, match="Duplicate"):
            for _ in ds._get_turns(
                [dialogue[0]["dialogue"][0]] + [dialogue[0]["dialogue"][0]]
            ):
                pass

        # Test incomplete turn
        with pytest.raises(ValueError, match="Incomplete turn in dialogue"):
            for _ in ds._get_turns(dialogue[0]["dialogue"][:2]):
                pass

    @mock.patch("torchtune.datasets._chat.load_dataset")
    def test_get_item(self, mock_load_dataset, template, dialogue):
        mock_load_dataset.return_value = dialogue
        expected_tokenized_prompts = [
            [
                0,
                7,
                3,
                3,
                2,
                2,
                10,
                5,
                4,
                2,
                3,
                7,
                2,
                5,
                10,
                3,
                7,
                2,
                4,
                2,
                3,
                -1,
                0,
                5,
                6,
                11,
                10,
                1,
                6,
                -1,
            ]
        ]
        prompt_lengths = (15, 5)
        expected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[0]
            + [
                3,
                7,
                2,
                4,
                2,
                3,
                -1,
            ]
            + [CROSS_ENTROPY_IGNORE_IDX] * prompt_lengths[1]
            + [1, 6, -1]
        ]

        ds = ChatDataset(
            tokenizer=DummyTokenizer(),
            source="iam/agoofy/goober",
            convert_to_dialogue=lambda x: x["dialogue"],
            template=template,
            max_seq_len=100,
            train_on_input=False,
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]
        assert prompt == expected_tokenized_prompts[0]
        assert label == expected_labels[0]
