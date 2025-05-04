
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from tests.common import ASSETS
from torchtune.data import Message
from torchtune.models.gemma import gemma_tokenizer


class TestGemmaTokenizer:
    @pytest.fixture
    def tokenizer(self):
        # m.model is a pretrained Sentencepiece model using the following command:
        # spm.SentencePieceTrainer.train('--input=<TRAIN_FILE> --model_prefix=m --vocab_size=2000')
        return gemma_tokenizer(str(ASSETS / "m.model"))

    @pytest.fixture
    def sample_messages(self):
        return [
            Message(
                role="user",
                content="Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n### Instruction:\nGenerate "
                "a realistic dating profile bio.\n\n### Response:\n",
                masked=True,
            ),
            Message(
                role="assistant",
                content="I'm an outgoing and friendly person who loves spending time with "
                "friends and family. I'm also a big-time foodie and love trying out new "
                "restaurants and different cuisines. I'm a big fan of the arts and enjoy "
                "going to museums and galleries. I'm looking for someone who shares my "
                "interest in exploring new places, as well as someone who appreciates a "
                "good conversation over coffee.",
                masked=False, # Masked = False for assistant as per common practice
            ),
        ]

    @pytest.fixture
    def token_lengths(self, tokenizer, sample_messages):
        """Helper fixture to calculate actual token lengths."""
        # Use .text_content to get the string needed by sentencepiece encode
        user_tokens = tokenizer.encode(sample_messages[0].text_content, add_bos=False, add_eos=False)
        asst_tokens = tokenizer.encode(sample_messages[1].text_content, add_bos=False, add_eos=False)
        user_len = len(user_tokens)
        asst_len = len(asst_tokens)
        # print(f"DEBUG: Calculated user_len={user_len}, asst_len={asst_len}") # Uncomment for debugging
        return user_len, asst_len

    # Expected tokens extracted from failing test run output
    @pytest.fixture
    def expected_tokens_with_eos(self):
        # fmt: off
        return [1, 323, 418, 202, 31, 128, 15, 120, 47, 88, 584, 23, 1665, 182, 9, 434, 295, 85, 4, 780, 47, 636, 9, 1094, 213, 23, 9, 69, 69, 164, 1153, 299, 35, 961, 132, 237, 7, 5, 761, 4, 12, 0, 313, 120, 47, 88, 584, 166, 493, 171, 54, 299, 9, 906, 244, 19, 186, 767, 303, 671, 92, 209, 24, 190, 52, 38, 4, 12, 0, 1243, 7, 69, 135, 213, 166, 6, 21, 45, 128, 71, 58, 38, 14, 10, 652, 35, 462, 101, 1306, 7, 341, 171, 20, 14, 127, 26, 652, 7, 10, 1268, 4, 6, 21, 45, 591, 9, 566, 22, 994, 913, 38, 20, 52, 24, 10, 1306, 734, 14, 71, 365, 1382, 7, 10, 801, 105, 88, 244, 985, 7, 4, 6, 21, 45, 9, 566, 126, 180, 11, 5, 1137, 7, 10, 1089, 151, 8, 1156, 213, 342, 7, 10, 384, 104, 54, 470, 4, 6, 21, 45, 287, 14, 33, 125, 135, 24, 101, 512, 66, 7, 28, 822, 15, 542, 69, 59, 110, 14, 365, 229, 7, 3, 36, 267, 36, 125, 135, 24, 101, 1503, 182, 9, 222, 1661, 191, 332, 92, 92, 24, 24, 4, 2]
        # fmt: on

    @pytest.fixture
    def expected_tokens_without_eos(self):
        # fmt: off
        return [1, 323, 418, 202, 31, 128, 15, 120, 47, 88, 584, 23, 1665, 182, 9, 434, 295, 85, 4, 780, 47, 636, 9, 1094, 213, 23, 9, 69, 69, 164, 1153, 299, 35, 961, 132, 237, 7, 5, 761, 4, 12, 0, 313, 120, 47, 88, 584, 166, 493, 171, 54, 299, 9, 906, 244, 19, 186, 767, 303, 671, 92, 209, 24, 190, 52, 38, 4, 12, 0, 1243, 7, 69, 135, 213, 166, 6, 21, 45, 128, 71, 58, 38, 14, 10, 652, 35, 462, 101, 1306, 7, 341, 171, 20, 14, 127, 26, 652, 7, 10, 1268, 4, 6, 21, 45, 591, 9, 566, 22, 994, 913, 38, 20, 52, 24, 10, 1306, 734, 14, 71, 365, 1382, 7, 10, 801, 105, 88, 244, 985, 7, 4, 6, 21, 45, 9, 566, 126, 180, 11, 5, 1137, 7, 10, 1089, 151, 8, 1156, 213, 342, 7, 10, 384, 104, 54, 470, 4, 6, 21, 45, 287, 14, 33, 125, 135, 24, 101, 512, 66, 7, 28, 822, 15, 542, 69, 59, 110, 14, 365, 229, 7, 3, 36, 267, 36, 125, 135, 24, 101, 1503, 182, 9, 222, 1661, 191, 332, 92, 92, 24, 24, 4]
        # fmt: on

    def test_tokenize_messages_add_end_tokens_true(self, tokenizer, sample_messages, token_lengths, expected_tokens_with_eos):
        user_len, asst_len = token_lengths
        tokens, mask = tokenizer.tokenize_messages(sample_messages, add_end_tokens=True)

        # Expected mask based on message.masked attribute and BOS/EOS
        # BOS=True, User=True (masked=True), Assistant=False (masked=False), EOS=True
        expected_mask = [True] * (1 + user_len) + [False] * asst_len + [True]

        # Direct comparison. Pytest will show diff on failure.
        assert tokens == expected_tokens_with_eos, f"Tokens mismatch."
        assert mask == expected_mask, f"Mask mismatch."
        assert len(tokens) == 1 + user_len + asst_len + 1
        assert len(mask) == 1 + user_len + asst_len + 1

    def test_tokenize_messages_add_end_tokens_false(self, tokenizer, sample_messages, token_lengths, expected_tokens_without_eos):
        user_len, asst_len = token_lengths
        tokens, mask = tokenizer.tokenize_messages(sample_messages, add_end_tokens=False)

        # Expected mask: BOS=True, User=True, Assistant=False
        expected_mask = [True] * (1 + user_len) + [False] * asst_len

        # Direct comparison. Pytest will show diff on failure.
        assert tokens == expected_tokens_without_eos, f"Tokens mismatch."
        assert mask == expected_mask, f"Mask mismatch."
        assert len(tokens) == 1 + user_len + asst_len
        assert len(mask) == 1 + user_len + asst_len

    def test_call_method_train(self, tokenizer, sample_messages, token_lengths, expected_tokens_with_eos):
        user_len, asst_len = token_lengths
        sample = {"messages": sample_messages}
        result = tokenizer(sample, inference=False) # add_end_tokens = True

        expected_mask = [True] * (1 + user_len) + [False] * asst_len + [True]

        assert "tokens" in result
        assert "mask" in result
        assert "messages" not in result
        # Direct comparison. Pytest will show diff on failure.
        assert result["tokens"] == expected_tokens_with_eos, f"Tokens mismatch."
        assert result["mask"] == expected_mask, f"Mask mismatch."
        assert len(result["tokens"]) == 1 + user_len + asst_len + 1
        assert len(result["mask"]) == 1 + user_len + asst_len + 1

    def test_call_method_inference(self, tokenizer, sample_messages, token_lengths, expected_tokens_without_eos):
        user_len, asst_len = token_lengths
        sample = {"messages": sample_messages}
        result = tokenizer(sample, inference=True) # add_end_tokens = False

        expected_mask = [True] * (1 + user_len) + [False] * asst_len

        assert "tokens" in result
        assert "mask" in result
        assert "messages" not in result
        # Direct comparison. Pytest will show diff on failure.
        assert result["tokens"] == expected_tokens_without_eos, f"Tokens mismatch."
        assert result["mask"] == expected_mask, f"Mask mismatch."
        assert len(result["tokens"]) == 1 + user_len + asst_len
        assert len(result["mask"]) == 1 + user_len + asst_len
