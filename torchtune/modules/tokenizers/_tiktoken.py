# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
from torchtune.modules.tokenizers._base import TokenEncoding
from torchtune.modules.tokenizers._utils import _split_long_repetitions

# Constants controlling encode logic
MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACE_CHARS = 25_000


class TikTokenEncoding(TokenEncoding):
    """A wrapper around tiktoken Encoding.

    Args:
        path (str): Path to pretrained tokenizer checkpoint file.
        name (str): Name of the tokenizer (used by tiktoken for identification).
        pattern (str): Regex pattern used to for string parsing.
        special_tokens (Optional[List[str]]): List of all special tokens. First element
            must be bos token, second element must be eos token, final element must be
            python tag. All elements must be unique. Length must be at most 256.
            Default: None (will use ALL_SPECIAL_TOKENS)
    """

    def __init__(
        self,
        path: str,
        name: str,
        pattern: str,
        special_tokens: Dict[str, int],
    ):
        mergeable_ranks = load_tiktoken_bpe(path)
        self.tt_model = Encoding(
            name=name,
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        # Vocab size without special tokens
        self.base_vocab_size = len(mergeable_ranks)
        # Vocab size with special tokens
        self.vocab_size = self.tt_model.n_vocab

    def encode(
        self,
        text: str,
        add_bos: bool,
        add_eos: bool,
    ) -> List[int]:
        """
        Encode a string into a list of token ids. Assumes that the string
        contains no special tokens.

        Args:
            text (str): The string to encode.
            add_bos (bool): Whether to add the beginning of sequence token.
            add_eos (bool): Whether to add the end of sequence token.

        Returns:
            List[int]: The list of token ids.
        """
        substrs: List[str] = []
        tokens = []
        if not text:
            return []
        for i in range(0, len(text), MAX_ENCODE_CHARS):
            substr = text[i : i + MAX_ENCODE_CHARS]
            # See https://github.com/openai/tiktoken/issues/195
            sliced_substr = _split_long_repetitions(substr, MAX_NO_WHITESPACE_CHARS)
            substrs.extend(sliced_substr)
        for substr in substrs:
            # allowed_special and disallowed_special are used by tiktoken to define
            # how special tokens are encoded. Our setting here is to encode any
            # special token as regular text and prevent tiktoken from raising errors.
            # This means we should only call encode on strings not containing special tokens.
            tokens.extend(
                self.tt_model.encode(
                    substr,
                    allowed_special=set(),
                    disallowed_special=(),
                )
            )
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token.

        Returns:
            str: The decoded string.
        """
        if truncate_at_eos:
            try:
                k = token_ids.index(self.eos_id)
            except ValueError:
                k = None
            if k:
                token_ids = token_ids[:k]
        token_ids = [token_id for token_id in token_ids if token_id != self.bos_id]
        return self.tt_model.decode(token_ids)
