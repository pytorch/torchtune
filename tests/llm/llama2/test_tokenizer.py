# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
import torch
from llm.llama2.tokenizer import Tokenizer

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return Tokenizer.from_file(str(ASSETS / "m.model"))

    def test_encode(self, tokenizer):
        assert tokenizer.encode("Hello world!") == [1, 12, 1803, 1024, 103, 2]
        assert tokenizer.encode("Hello world!", add_eos=False) == [
            1,
            12,
            1803,
            1024,
            103,
        ]
        assert tokenizer.encode("Hello world!", add_bos=False) == [
            12,
            1803,
            1024,
            103,
            2,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False, add_bos=False) == [
            12,
            1803,
            1024,
            103,
        ]
        assert tokenizer.encode(["Hello world!"]) == [[1, 12, 1803, 1024, 103, 2]]
        assert tokenizer.encode(["Hello world!", "what"]) == [
            [1, 12, 1803, 1024, 103, 2],
            [1, 121, 2],
        ]
        assert torch.allclose(
            tokenizer.encode("Hello world!", return_as_tensor=True),
            torch.tensor([1, 12, 1803, 1024, 103, 2]),
        )
        assert torch.allclose(
            tokenizer.encode(["Hello world!", "what"], return_as_tensor=True),
            torch.tensor([[1, 12, 1803, 1024, 103, 2], [1, 121, 2, -1, -1, -1]]),
        )

    def test_decode(self, tokenizer):
        assert tokenizer.decode([1, 12, 1803, 1024, 103, 2]) == ["Hello world!"]
        assert tokenizer.decode([[1, 12, 1803, 1024, 103, 2], [1, 121, 2]]) == [
            "Hello world!",
            "what",
        ]
        assert tokenizer.decode(torch.tensor([1, 12, 1803, 1024, 103, 2])) == [
            "Hello world!"
        ]
        assert tokenizer.decode(
            torch.tensor([[1, 12, 1803, 1024, 103, 2], [1, 121, 2, -1, -1, -1]])
        ) == ["Hello world!", "what"]
