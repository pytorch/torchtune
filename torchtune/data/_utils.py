# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional


def truncate(
    tokens: List[Any],
    max_seq_len: int,
    eos_id: Optional[Any] = None,
) -> List[Any]:
    """
    Truncate a list of tokens to a maximum length. If eos_id is provided, the last
    token will be replaced with eos_id.

    Args:
        tokens (List[Any]): list of tokens to truncate
        max_seq_len (int): maximum length of the list
        eos_id (Optional[Any]): token to replace the last token with. If None, the
            last token will not be replaced. Default is None.

    Returns:
        List[Any]: truncated list of tokens
    """
    tokens_truncated = tokens[:max_seq_len]
    if eos_id is not None and tokens_truncated[-1] != eos_id:
        tokens_truncated[-1] = eos_id
    return tokens_truncated


def split_text_by_image_tag(content: str, image_tag: str) -> List[Dict[str, str]]:
    """
    Given a raw text string, split by the specified ``image_tag``
    and form into list of dictionaries to be used in the ``Message`` content
    field.

    Args:
        content (str): raw message text
        image_tag (str): string to split the text by

    Returns:
        List[Dict[str, str]]: list of dictionaries to be used in the ``Message`` content field

    Example:
        >>> content = split_text_by_image_tag("<image>hello <image>world", "<image>")
        >>> print(content)
        [{"type": "image"}, {"type": "text", "content": "hello "}, {"type": "image"}, {"type": "text", "content": "world"}]
    """
    split_content = content.split(image_tag)
    final_content_list = []
    for i, substr in enumerate(split_content):
        if len(substr) > 0:
            final_content_list.append({"type": "text", "content": substr})
        if i < len(split_content) - 1:
            final_content_list.append({"type": "image"})

    return final_content_list
