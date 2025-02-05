# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.common import ASSETS
from tokenizers import Tokenizer
from torchtune.models.llama3._tokenizer import CL100K_PATTERN
from torchtune.modules.transforms.tokenizers import HFTokenizer, TikTokenBaseTokenizer


TOKENIZER_CONFIG_PATH = ASSETS / "tokenizer_config.json"
GENERATION_CONFIG_PATH = ASSETS / "generation_config.json"


class TestHFTokenizer:
    @pytest.fixture
    def tt_tokenizer(self):
        # Pretrained tiktoken model generated via the script in
        # https://gist.github.com/ebsmothers/54b133dd87db6679b14318545aaa2de4
        return TikTokenBaseTokenizer(
            path=str(ASSETS / "tiktoken_small.model"),
            name="test_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=0,
            eos_id=-1,
            special_tokens={
                "<|test_token_0|>": 2000,
                "<|test_token_1|>": 2001,
            },
        )

    @pytest.fixture
    def texts(self):
        return [
            "I can see the sun. But even if I cannot see the sun, I know that it exists.",
            "And to know that the sun is there - that is living.",
        ]

    @pytest.fixture
    def token_ids(self):
        return [
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
        ]

    def test_invalid_hf_tokenizer(self):
        with pytest.raises(ValueError, match="Could not infer"):
            _ = HFTokenizer(
                path=str(ASSETS / "tokenizer.json"),
            )

    @pytest.mark.parametrize(
        "config_path, generation_config_path",
        [
            (TOKENIZER_CONFIG_PATH, GENERATION_CONFIG_PATH),
            (TOKENIZER_CONFIG_PATH, None),
            (None, GENERATION_CONFIG_PATH),
        ],
    )
    def test_tokenizer_encode_and_decode_parity(
        self,
        tt_tokenizer,
        texts,
        token_ids,
        config_path,
        generation_config_path,
        mocker,
    ):

        # Patch tokenizer's token_to_id method for BOS and EOS
        # since they are not present in the original tokenizer model
        def patch_token_to_id_for_dummy_tokenizer(*args, **kwargs):
            if args[0] == "<|begin_of_sentence|>":
                return 0
            elif args[0] == "<|end_of_sentence|>":
                return -1
            else:
                raise RuntimeError("Unexpected input")

        mocker.patch.object(
            Tokenizer, "token_to_id", side_effect=patch_token_to_id_for_dummy_tokenizer
        )
        # Tokenizer artifacts for this test were created from tiktoken_small.model
        # using the script in https://gist.github.com/ebsmothers/55b2f177f5ed15a3b81508f8f8b91159
        hf_tokenizer = HFTokenizer(
            path=str(ASSETS / "tokenizer.json"),
            config_path=config_path,
            generation_config_path=generation_config_path,
        )

        tt_tokens = tt_tokenizer.encode(texts[0], add_bos=True, add_eos=True)
        hf_tokens = hf_tokenizer.encode(texts[0], add_bos=True, add_eos=True)

        assert tt_tokens == hf_tokens
        assert hf_tokens == [0] + token_ids + [-1]

        tt_text = tt_tokenizer.decode(token_ids)
        hf_text = hf_tokenizer.decode(token_ids)
        assert tt_text == hf_text
        assert hf_text == texts[0]
