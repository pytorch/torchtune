# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Protocol, Tuple

from torchtune.data._types import Message


class TokenEncoding(Protocol):
    """
    Abstract token encoding model that implements ``encode`` and ``decode`` methods.
    """

    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.

        Returns:
            List[int]: The encoded list of token ids.
        """
        pass

    def decode(
        self, token_ids: List[int], include_special: bool = False, **kwargs
    ) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.
            include_special (bool): Whether to include special tokens in the decoded text.
                Default is False.

        Returns:
            str: The decoded text.
        """
        pass


class Tokenizer(Protocol):
    """
    Abstract tokenizer that implements model specific special token logic in
    ``tokenize_message`` and ``tokenize_messages`` methods.
    """

    def tokenize_messages(
        self, messages: List[Message], **kwargs
    ) -> Tuple[List[int], List[bool]]:
        """
        Given a list of messages, return a list of tokens and list of masks for
        the concatenated and formatted messages.

        Args:
            messages (List[Message]): The list of messages to tokenize.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        pass
