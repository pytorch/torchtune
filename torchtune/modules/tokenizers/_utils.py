# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional, Protocol, Tuple

from torchtune.data._messages import Message
from torchtune.data._utils import truncate


class BaseTokenizer(Protocol):
    """
    Abstract token encoding model that implements ``encode`` and ``decode`` methods.
    See :class:`~torchtune.modules.tokenizers.SentencePieceBaseTokenizer` and
    :class:`~torchtune.modules.tokenizers.TikTokenBaseTokenizer` for example implementations of this protocol.
    """

    def encode(self, text: str, **kwargs: Dict[str, Any]) -> List[int]:
        """
        Given a string, return the encoded list of token ids.

        Args:
            text (str): The text to encode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            List[int]: The encoded list of token ids.
        """
        pass

    def decode(self, token_ids: List[int], **kwargs: Dict[str, Any]) -> str:
        """
        Given a list of token ids, return the decoded text, optionally including special tokens.

        Args:
            token_ids (List[int]): The list of token ids to decode.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            str: The decoded text.
        """
        pass


class ModelTokenizer(Protocol):
    """
    Abstract tokenizer that implements model-specific special token logic in
    the ``tokenize_messages`` method. See :class:`~torchtune.models.llama3.Llama3Tokenizer`
    for an example implementation of this protocol.
    """

    special_tokens: Dict[str, int]
    max_seq_len: Optional[int]

    def tokenize_messages(
        self, messages: List[Message], **kwargs: Dict[str, Any]
    ) -> Tuple[List[int], List[bool]]:
        """
        Given a list of messages, return a list of tokens and list of masks for
        the concatenated and formatted messages.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            **kwargs (Dict[str, Any]): kwargs.

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        pass


def tokenize_messages_no_special_tokens(
    tokenizer: ModelTokenizer,
    messages: List[Message],
    *,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[int], List[bool]]:
    r"""Tokenize a list of messages one at a time then concatenate them,
    returning a list of tokens and a list of masks. Does not add any special
    tokens except for BOS and EOS (if provided). This serves as a common starting point for
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
        ...     bos_id=tokenizer.bos_id,
        ...     eos_id=tokenizer.eos_id,
        ... )[0]
        >>> print(tokens)
        [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]
        # Same result as encoding the full string in one go
        >>> print(tokenizer.encode(''.join([message.content for message in messages])))
        [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


    Args:
        tokenizer (ModelTokenizer): Tokenizer to encode messages with.
        messages (List[Message]): A list of messages, each containing role, content,
            and masked attributes.
        bos_id (Optional[int]): Beginning-of-sequence token id. If None, no BOS token will
            be added. Default None.
        eos_id (Optional[int]): End-of-sequence token id. If None, no EOS token will be added. Default None.

    Returns:
        Tuple[List[int], List[bool]]: The tokenized messages.

    Raises:
        RuntimeError: if any message in ``messages`` does not satisfy ``message['type'] == 'text'``.
    """
    start_of_turn = True
    end_of_turn = False
    prev_ends_with_space = False
    max_seq_len = tokenizer.max_seq_len  # We define this on ModelTokenizer
    tokenized_messages = []
    mask = []
    for message in messages:
        # If assistant message, this is the end of a turn
        end_of_turn = message.role == "assistant"

        # Prepend BOS on start of new turns
        if start_of_turn and bos_id is not None:
            tokenized_messages.append(bos_id)
            mask.append(message.masked)

        # We want to trim leading whitespace on the next message when
        # (a) it is a continuation of the turn (i.e. not the first message)
        # (b) the vocabulary explicitly encodes whitespace characters (checked inside
        #     the base tokenizer's encode method), and
        # (c) the previous message did not end with a space
        trim_leading_whitespace = (not start_of_turn) and not prev_ends_with_space

        # Tokenize current message, append with masks
        tokens = []
        for item in message.content:
            if item["type"] == "text":
                tokens = tokens + tokenizer.encode(
                    item["content"].rstrip(" "),
                    add_bos=False,
                    add_eos=False,
                    trim_leading_whitespace=trim_leading_whitespace,
                )
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")
        prev_ends_with_space = item["content"].endswith(" ")
        tokenized_messages.extend(tokens)
        mask.extend([message.masked] * len(tokens))

        # If assistant message, append EOS at end
        if end_of_turn:
            if eos_id is not None:
                tokenized_messages.append(eos_id)
                mask.append(message.masked)
            end_of_turn = False
            start_of_turn = True
        else:
            start_of_turn = False

        # Break out early if we reach max_seq_len
        if max_seq_len is not None and len(tokenized_messages) >= max_seq_len:
            break

    # Finally, truncate if necessary
    if max_seq_len is not None:
        tokenized_messages = truncate(tokenized_messages, max_seq_len, eos_id)
        mask = truncate(
            mask, max_seq_len, message.masked if eos_id is not None else None
        )

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
