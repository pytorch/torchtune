# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Protocol

from torchtune.data._types import Message


class TokenEncoding(Protocol):
    """
    Abstract token encoding model that implements ``encode`` and ``decode`` methods.
    """

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Given a string, return the a list of token ids.
        """
        pass

    def decode(
        self, token_ids: List[int], include_special: bool = False, **kwargs
    ) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.
        """
        pass


class Tokenizer(Protocol):
    """
    Abstract tokenizer that implements model specific special token logic in
    ``tokenize_message`` and ``tokenize_messages`` methods.
    """

    def tokenize_messages(self, messages: List[Message], **kwargs):
        """
        Given a list of messages, return a list of tokens for the concatenated
        and formatted messages.
        """
        pass
