# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
from torchtune.modules.tokenizer import Tokenizer

ASSETS = Path(__file__).parent.parent.parent / "assets"


class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(ASSETS / "m.model"))

    @pytest.fixture
    def tokenizer_with_max_token_len(self, request):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return Tokenizer.from_file(str(ASSETS / "m.model"), request.param)

    # Parameters - 2 = too small for truncation, truncate, full token list
    @pytest.mark.parametrize(
        "tokenizer_with_max_token_len",
        [2, 4, 100],
        indirect=["tokenizer_with_max_token_len"],
    )
    def test_truncation(self, tokenizer_with_max_token_len):
        full_token_list = [
            tokenizer_with_max_token_len.bos_id,
            12,
            1803,
            1024,
            103,
            tokenizer_with_max_token_len.eos_id,
        ]

        truncated_token_list = [
            tokenizer_with_max_token_len.bos_id,
            12,
            1803,
            tokenizer_with_max_token_len.eos_id,
        ]

        token_list = (
            truncated_token_list
            if tokenizer_with_max_token_len.max_token_len == 4
            else full_token_list
        )
        assert (
            tokenizer_with_max_token_len.encode("Hello world!", truncate=True)
            == token_list
        )

    @pytest.mark.parametrize(
        "tokenizer_with_max_token_len", [1], indirect=["tokenizer_with_max_token_len"]
    )
    def test_truncation_without_bos_eos(self, tokenizer_with_max_token_len):
        truncated_token_list = [12]
        assert (
            tokenizer_with_max_token_len.encode(
                "Hello world!", add_bos=False, add_eos=False, truncate=True
            )
            == truncated_token_list
        )

    def test_truncation_param_error(self):
        with pytest.raises(ValueError):
            Tokenizer.from_file(str(ASSETS / "m.model"), 0)
        with pytest.raises(ValueError):
            Tokenizer.from_file(str(ASSETS / "m.model"), -5)

    @pytest.mark.parametrize(
        "tokenizer_with_max_token_len",
        [1, 2, 4, 100],
        indirect=["tokenizer_with_max_token_len"],
    )
    def test_truncation_without_eos(self, tokenizer_with_max_token_len):
        full_token_list = [
            tokenizer_with_max_token_len.bos_id,
            12,
            1803,
            1024,
            103,
        ]

        larger_truncated_token_list = [
            tokenizer_with_max_token_len.bos_id,
            12,
            1803,
            1024,
        ]

        smaller_truncated_token_list = [
            tokenizer_with_max_token_len.bos_id,
            12,
        ]

        if (
            tokenizer_with_max_token_len.max_token_len == 100
            or tokenizer_with_max_token_len.max_token_len == 1
        ):
            token_list = full_token_list
        elif tokenizer_with_max_token_len.max_token_len == 4:
            token_list = larger_truncated_token_list
        else:
            token_list = smaller_truncated_token_list

        assert (
            tokenizer_with_max_token_len.encode(
                "Hello world!", add_eos=False, truncate=True
            )
            == token_list
        )

    def test_encode(self, tokenizer):
        assert tokenizer.encode("Hello world!") == [
            tokenizer.bos_id,
            12,
            1803,
            1024,
            103,
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False) == [
            tokenizer.bos_id,
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
            tokenizer.eos_id,
        ]
        assert tokenizer.encode("Hello world!", add_eos=False, add_bos=False) == [
            12,
            1803,
            1024,
            103,
        ]

    def test_decode(self, tokenizer):
        assert tokenizer.decode([1, 12, 1803, 1024, 103, 2]) == "Hello world!"

    def test_token_ids(self, tokenizer):
        assert tokenizer.eos_id == 2
        assert tokenizer.pad_id == -1
        assert tokenizer.bos_id == 1

    def test_tokenizer_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 2000
