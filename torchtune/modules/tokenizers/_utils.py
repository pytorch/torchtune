# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, List, Protocol, Union

from torchtune.data._types import Message


class Tokenizer(Protocol):
    """Abstract tokenizer"""

    bos_id: int
    eos_id: int
    pad_id: int

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Given a string, return the a list of token ids.
        """

    def decode(
        self, token_ids: List[int], add_bos: bool, add_eos: bool, **kwargs
    ) -> str:
        """
        Given a list of token ids, return the decoded text.
        """

    def tokenize_messages(self, token_ids: List[Message], **kwargs):
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.
        """
        pass


def truncate(
    tokens: List[int],
    max_seq_len: int,
    eos_id: Union[int, bool],
):
    tokens_truncated = tokens[:max_seq_len]
    if tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated


def _split_long_repetitions(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
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
