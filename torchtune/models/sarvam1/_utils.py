# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Protocol, Tuple

from torchtune.data._messages import Message
from torchtune.data._utils import truncate
from torchtune.modules.tokenizers._utils import ModelTokenizer

# https://github.com/pytorch/torchtune/blob/26b2200010a37474015925c5e3f4606435b72dd3/torchtune/modules/tokenizers/_utils.py#L75


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

        # Tokenize current message, append with masks
        tokens = []
        for item in message.content:
            if item["type"] == "text":
                tokens = tokens + tokenizer.encode(
                    item["content"],
                    add_bos=False,
                    add_eos=False,
                    trim_leading_whitespace=True,
                )
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")
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
