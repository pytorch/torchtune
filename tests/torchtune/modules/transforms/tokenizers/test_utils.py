# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from unittest.mock import patch, MagicMock

from tests.test_utils import DummyTokenizer
from torchtune.data import Message

from torchtune.modules.transforms.tokenizers import tokenize_messages_no_special_tokens

from torchtune.modules.transforms.tokenizers import has_trainable_tokens
from torchtune.utils._logging import get_logger

class TestTokenizerUtils:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer(max_seq_len=100)

    @pytest.fixture
    def messages(self):
        return [
            Message(role="user", content="hello world!", masked=True),
            Message(role="assistant", content="hello back!"),
        ]

    @pytest.mark.parametrize(
        "add_bos, add_eos",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_tokenize_no_special_tokens(self, tokenizer, messages, add_bos, add_eos):
        tokens, mask = tokenize_messages_no_special_tokens(
            tokenizer,
            messages,
            bos_id=tokenizer.bos_id if add_bos else None,
            eos_id=tokenizer.eos_id if add_eos else None,
        )

        assert len(tokens) == len(mask)

        # User message should be masked
        assert mask[0] is True
        # Assistant message should not be masked
        assert mask[-1] is False

        if add_bos:
            assert tokens[0] == tokenizer.bos_id
        else:
            assert tokens[0] != tokenizer.bos_id

        if add_eos:
            assert tokens[-1] == tokenizer.eos_id
        else:
            assert tokens[-1] != tokenizer.eos_id

class TestHasTrainableTokens:
    def test_all_ignore_index(self):
        ignore_index = -100
        labels = torch.tensor([ignore_index, ignore_index])
        result = has_trainable_tokens(labels, ignore_index)
        assert result is False

    def test_some_trainable_tokens(self):
        ignore_index = -100
        labels = torch.tensor([ignore_index, 0])
        result = has_trainable_tokens(labels, ignore_index)
        assert result is True

    def test_none_labels(self):
        result = has_trainable_tokens(None, -100)
        assert result is False

    def test_empty_labels_tensor(self):
        labels = torch.tensor([])
        result = has_trainable_tokens(labels, -100)
        assert result is False

    @patch('torchtune.modules.transforms.tokenizers._utils.log_once')
    def test_logging(self, mock_log_once):
        ignore_index = -100
        labels = torch.tensor([ignore_index, ignore_index])
        _ = has_trainable_tokens(labels, ignore_index)
        mock_log_once.assert_called_once_with(
            get_logger(),
            'Consider changing to tokenizer.truncation="left" or increase tokernizer.max_seq_len.'
        )