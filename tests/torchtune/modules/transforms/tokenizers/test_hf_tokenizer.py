# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import pytest
from tests.common import ASSETS
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from torchtune.data import Message
from torchtune.models.llama3._tokenizer import CL100K_PATTERN
from torchtune.modules.transforms.tokenizers import (
    HuggingFaceBaseTokenizer,
    HuggingFaceModelTokenizer,
    TikTokenBaseTokenizer,
)

TOKENIZER_CONFIG_PATH = ASSETS / "tokenizer_config.json"
GENERATION_CONFIG_PATH = ASSETS / "generation_config.json"


class TestHuggingFaceBaseTokenizer:
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
        with pytest.raises(ValueError, match="At least one of"):
            _ = HuggingFaceBaseTokenizer(
                tokenizer_json_path=str(ASSETS / "tokenizer.json"),
            )

    @pytest.mark.parametrize(
        "config_path, generation_config_path, hf_tokenizer_adds_bos",
        [
            (TOKENIZER_CONFIG_PATH, GENERATION_CONFIG_PATH, True),
            (TOKENIZER_CONFIG_PATH, None, False),
            (None, GENERATION_CONFIG_PATH, True),
        ],
    )
    def test_tokenizer_encode_and_decode_parity(
        self,
        tt_tokenizer,
        texts,
        token_ids,
        config_path,
        generation_config_path,
        hf_tokenizer_adds_bos,
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
        hf_tokenizer = HuggingFaceBaseTokenizer(
            tokenizer_json_path=str(ASSETS / "tokenizer.json"),
            tokenizer_config_json_path=config_path,
            generation_config_path=generation_config_path,
        )

        if hf_tokenizer_adds_bos:
            # This is a hacky way to patch the post-processor to prepend BOS
            # (Patching with mocker doesn't work)
            post_processor = TemplateProcessing(
                single="<BOS> $0", pair="<BOS> $A $B", special_tokens=[("<BOS>", 0)]
            )
            hf_tokenizer.tokenizer.post_processor = post_processor
            # Validate that the patch worked
            assert hf_tokenizer.tokenizer.encode("").ids == [0]
            # Re-call the method with the new post-processor
            hf_tokenizer._infer_should_add_bos_eos()

        tt_tokens = tt_tokenizer.encode(texts[0], add_bos=True, add_eos=True)
        hf_tokens = hf_tokenizer.encode(texts[0], add_bos=True, add_eos=True)

        assert tt_tokens == hf_tokens
        assert hf_tokens == [0] + token_ids + [-1]

        tt_text = tt_tokenizer.decode(token_ids)
        hf_text = hf_tokenizer.decode(token_ids)
        assert tt_text == hf_text
        assert hf_text == texts[0]


GEMMA_TOKENIZER_CONFIG_PATH = ASSETS / "tokenizer_config_gemma.json"
GEMMA_GENERATION_CONFIG_PATH = ASSETS / "generation_config_gemma.json"
GEMMA_TOKENIZER_PATH = ASSETS / "tokenizer_gemma_cropped.json"

QWEN2_5_TOKENIZER_CONFIG_PATH = ASSETS / "qwen2_5-tokenizer/tokenizer_config.json"
QWEN2_5_TOKENIZER_PATH = ASSETS / "qwen2_5-tokenizer/tokenizer_cropped.json"


class TestHuggingFaceModelTokenizer:
    """
    Tokenizer asset for this test was generated using https://gist.github.com/krammnic/a80602f5bff03921096876c995351da8
    """

    @pytest.fixture
    def model_tokenizer(self):
        return HuggingFaceModelTokenizer(
            tokenizer_json_path=str(GEMMA_TOKENIZER_PATH),
            tokenizer_config_json_path=str(GEMMA_TOKENIZER_CONFIG_PATH),
            generation_config_path=str(GEMMA_GENERATION_CONFIG_PATH),
        )

    @pytest.fixture
    def tool_call_tokenizer(self):
        return HuggingFaceModelTokenizer(
            tokenizer_json_path=str(QWEN2_5_TOKENIZER_PATH),
            tokenizer_config_json_path=str(QWEN2_5_TOKENIZER_CONFIG_PATH),
            generation_config_path=None,
        )

    @pytest.fixture
    def messages(self):
        return [
            Message(
                role="user",
                content="hello there",
                masked=False,
                tool_calls=[],
            ),
            Message(
                role="assistant",
                content="hi",
                masked=False,
                tool_calls=[],
            ),
            Message(
                role="user",
                content="whatsup?",
                masked=False,
                tool_calls=[],
            ),
        ]

    @pytest.fixture
    def tool_call_messages(self):
        messages = [
            Message(
                role="user",
                content="I maintain that it is the suffering of being unable to love.",
                masked=True,
                eot=True,
                tool_calls=False,
            ),
            Message(
                role="assistant",
                content="What is hell? ",
                masked=True,
                eot=True,
                tool_calls=[
                    {
                        "name": "get_current_temperature",
                        "arguments": '{"location": "Hell 9th Circle, Cocytus", "unit": "celsius"}',
                    }
                ],
            ),
        ]
        return messages

    def test_no_mask(self, model_tokenizer, messages):
        tokens, mask = model_tokenizer.tokenize_messages(messages)

        assert tokens[:-4] == [
            2,
            4,
            53,
            74,
            71,
            14,
            5,
            6,
            4,
            51,
            50,
            49,
            17,
            5,
            6,
            4,
            82,
            22,
            9,
            5,
            6,
            4,
        ]

        assert mask[:-4] == [False] * 22

    def test_with_mask(self, model_tokenizer, messages):
        """
        In this test we mask the first message and verify that it does not affect tokens,
        and message mask is changed in the correct way.
        """
        messages[0].masked = True
        tokens, mask = model_tokenizer.tokenize_messages(messages)

        assert tokens[:-4] == [
            2,
            4,
            53,
            74,
            71,
            14,
            5,
            6,
            4,
            51,
            50,
            49,
            17,
            5,
            6,
            4,
            82,
            22,
            9,
            5,
            6,
            4,
        ]

        assert mask[:-4] == [True] * 8 + [False] * 14

    def test_with_tools(self, tool_call_tokenizer, tool_call_messages):
        tokens, mask = tool_call_tokenizer.tokenize_messages(tool_call_messages)

        assert tokens == [
            151644,
            8948,
            198,
            2610,
            525,
            1207,
            16948,
            11,
            3465,
            553,
            54364,
            14817,
            13,
            1446,
            525,
            264,
            10950,
            17847,
            13,
            151645,
            198,
            151644,
            872,
            198,
            40,
            10306,
            429,
            432,
            374,
            279,
            15691,
            315,
            1660,
            11889,
            311,
            2948,
            13,
            151645,
            198,
            151644,
            77091,
            198,
            3838,
            374,
            14780,
            30,
            715,
            151657,
            198,
            4913,
            606,
            788,
            330,
            455,
            11080,
            53525,
            497,
            330,
            16370,
            788,
            54734,
            2527,
            11693,
            7245,
            80294,
            220,
            24,
            339,
            21224,
            11,
            76388,
            16415,
            355,
            16215,
            7245,
            3843,
            11693,
            7245,
            66,
            40247,
            2105,
            9863,
            532,
            151658,
            151645,
            198,
            151644,
            77091,
            198,
            151645,
        ]

        assert mask == [True] * 90
