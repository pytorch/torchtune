# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Dict, List, Optional, Protocol, Tuple

from torchtune.data._types import Message
from torchtune.data._utils import truncate


class BaseTokenizer(Protocol):
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

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.

        Returns:
            str: The decoded text.
        """
        pass


class ModelTokenizer(Protocol):
    """
    Abstract tokenizer that implements model specific special token logic in
    the ``tokenize_messages`` method.
    """

    special_tokens: Dict[str, int]

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


def tokenize_messages_no_special_tokens(
    tokenizer: ModelTokenizer,
    messages: List[Message],
    bos_id: int,
    eos_id: int,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[int], List[bool]]:
    r"""Tokenize a list of messages one at a time then concatenate them,
    returning a list of tokens and a list of masks. Does not add any special
    tokens except for BOS and EOS. This serves as a common starting point for
    model tokenizers that do not rely heavily on special tokens.

    Examples:
        >>> messages = [
        ...     Message(role="system", content="system message\n", masked=True),
        ...     Message(role="user", content="user prompt\n", masked=True),
        ...     Message(role="assistant", content="assistant response\n"),
        ... ]
        # tokenize_messages encodes messages separately and concats
        >>> tokens = tokenize_messages_no_special_tokens(
        ...     tokenizer,
        ...     messages,
        ...     tokenizer.bos_id,
        ...     tokenizer.eos_id,
        ...     max_seq_len
        ... )[0]
        >>> print(tokens)
        [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]
        # Same result as encoding the full string in one go
        >>> print(tokenizer.encode(''.join([message.content for message in messages])))
        [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


    Args:
        messages (List[Message]): A list of messages, each containing role, content,
            and masked attributes.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None

    Returns:
        Tuple[List[int], List[bool]]: The tokenized messages
    """
    start_of_turn = True
    end_of_turn = False
    prev_ends_with_space = False
    tokenized_messages = []
    mask = []
    for message in messages:
        # If assistant message, this is the end of a turn
        end_of_turn = message.role == "assistant"

        # Prepend BOS on start of new turns
        if start_of_turn:
            tokenized_messages.append(bos_id)
            mask.append(message.masked)

        # We want to trim leading whitespace on the next message when
        # (a) it is a continuation of the turn (i.e. not the first message)
        # (b) the vocabulary explicitly encodes whitespace characters (checked inside
        #     the base tokenizer's encode method), and
        # (c) the previous message did not end with a space
        trim_leading_whitespace = (not start_of_turn) and not prev_ends_with_space

        # Tokenize current message, append with masks
        tokens = tokenizer.encode(
            message.content.rstrip(" "),
            add_bos=False,
            add_eos=False,
            trim_leading_whitespace=trim_leading_whitespace,
        )
        prev_ends_with_space = message.content.endswith(" ")
        tokenized_messages.extend(tokens)
        mask.extend([message.masked] * len(tokens))

        # If assistant message, append EOS at end
        if end_of_turn:
            tokenized_messages.append(eos_id)
            mask.append(message.masked)
            end_of_turn = False
            start_of_turn = True
        else:
            start_of_turn = False

        # Break out early if we reach max_seq_len
        if max_seq_len and len(tokenized_messages) >= max_seq_len:
            break

    # Finally, truncate if necessary
    if max_seq_len:
        tokenized_messages = truncate(tokenized_messages, max_seq_len, eos_id)
        mask = truncate(mask, max_seq_len, message.masked)

    return tokenized_messages, mask


def parse_hf_tokenizer_json(tokenizer_json_path: str) -> Dict[str, int]:
    """
    Parse the ``tokenizer.json`` file from a Hugging Face model to extract the
    special token str to id mapping.

    Args:
        tokenizer_json_path (str): Path to the ``tokenizer.json`` file.

    Returns:
        Dict[str, int]: The special token str to id mapping.
    """
    with open(tokenizer_json_path, "r") as f:
        tokenizer_json = json.load(f)

    return {token["content"]: token["id"] for token in tokenizer_json["added_tokens"]}
