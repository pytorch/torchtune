# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, TypeVar

from torchtune.config._utils import _get_component_from_path

from torchtune.data._messages import Message
from torchtune.data._prompt_templates import (
    _TemplateType,
    PromptTemplate,
    PromptTemplateInterface,
)

T = TypeVar("T", bound=type)


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
    field::

        [
            {
                "role": "system" | "user" | "assistant",
                "content":
                    [
                        {"type": "image"},
                        {"type": "text", "content": "This is a sample image."},
                    ],
            },
            ...
        ]

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


def validate_messages(
    messages: List[Message],
) -> None:
    """
    Given a list of messages, ensure that messages form a valid
    back-and-forth conversation. An error will be raised if:

    - There is a system message that's not the first message
    - There are two consecutive user messages
    - An assistant message comes before the first user message
    - The message is empty
    - Messages are shorter than length of 2 (min. one user-assistant turn)


    Args:
        messages (List[Message]): the messages to validate.

    Raises:
        ValueError: If the messages are invalid.
    """
    if len(messages) < 2:
        raise ValueError(
            f"Messages must be at least length 2, but got {len(messages)} messages"
        )

    last_turn = "assistant"
    for i, message in enumerate(messages):
        if message.role == "assistant" and last_turn != "user":
            raise ValueError(
                f"Assistant message before expected user message at index {i} in messages"
            )
        if message.role == "user" and last_turn == "user":
            raise ValueError(
                f"Two consecutive user messages at index {i} and {i - 1} in messages"
            )
        if message.role == "system" and i > 0:
            raise ValueError(
                f"System message at index {i} in messages, but system messages must come first"
            )
        last_turn = message.role


def _get_prompt_template(
    prompt_template: _TemplateType,
) -> PromptTemplateInterface:
    """
    Retrieve prompt template from import dotpath or create a custom one with provided
    template dictionary.

    Args:
        prompt_template (_TemplateType): optional specified prompt template.
            If a string, it is assumed to be the dotpath of a :class:`~torchtune.data.PromptTemplateInterface`
            class. If a dictionary, it is assumed to be a custom prompt template mapping role to the
            prepend/append tags.

    Returns:
        PromptTemplateInterface: the specified prompt template

    Raises:
        ValueError: If a string or dictionary is not passed in
    """
    if isinstance(prompt_template, str):
        return _get_component_from_path(prompt_template)()
    elif isinstance(prompt_template, dict):
        return PromptTemplate(prompt_template)
    else:
        raise ValueError(
            f"Prompt template must be a dotpath string or dictionary with custom template, got {type(prompt_template)}"
        )
