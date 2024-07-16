# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterator, List

from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
from torchtune.modules.tokenizers._utils import BaseTokenizer

# Constants controlling encode logic
MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACE_CHARS = 25_000


class TikTokenBaseTokenizer(BaseTokenizer):
    """
    A lightweight wrapper around tiktoken Encoding. This class additionally handles
    breaking up the input text into substrings of a max length and splitting up long
    repetitions to improve encode speed.

    Args:
        path (str): Path to pretrained tokenizer checkpoint file.
        name (str): Name of the tokenizer (used by tiktoken for identification).
        pattern (str): Regex pattern used to split input text into chunks before passing
            to byte-pair encoding.
        bos_id (int): beginning-of-sequence token id. This can be present or absent in ``special_tokens``.
        eos_id (int): end-of-sequence token id. This can be present or absent in ``special_tokens``.
        special_tokens (Dict[str, int]): Mapping of special tokens to their ids.

    Examples:
        >>> tokenizer = TikTokenBaseTokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        name: str,
        pattern: str,
        bos_id: int,
        eos_id: int,
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
        self.bos_id = bos_id
        self.eos_id = eos_id

    def _split_long_repetitions(
        self, s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Split the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """
        Encode a string into a list of token ids. Assumes that the string
        contains no special tokens.

        Args:
            text (str): The string to encode.

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
            sliced_substr = self._split_long_repetitions(
                substr, MAX_NO_WHITESPACE_CHARS
            )
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
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
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
                sequence token. Default is True.

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
