# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.data import Message, split_text_by_image_tag, truncate, validate_messages


def test_truncate():
    tokens = [1, 2, 3, 4, -1]

    # Test no truncation
    truncated_tokens = truncate(
        tokens=tokens,
        max_seq_len=5,
        eos_id=-1,
    )
    assert truncated_tokens == tokens

    masks = [True, True, False, True, False]
    # Test truncated mask
    truncated_masks = truncate(tokens=masks, max_seq_len=4, eos_id=False)
    assert truncated_masks == [True, True, False, False]


def test_validate_messages():
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]

    # Test valid conversation with system
    validate_messages(messages)

    # Test valid conversation without system
    validate_messages(messages[1:])

    # Test system not first
    messages = [
        Message(role="user", content="hello"),
        Message(role="system", content="hello"),
        Message(role="assistant", content="world"),
    ]
    with pytest.raises(
        ValueError,
        match="System message at index 1 in messages, but system messages must come first",
    ):
        validate_messages(messages)

    # Test empty assistant message
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="world"),
        Message(role="assistant", content=""),
    ]
    validate_messages(messages)

    # Test single message
    messages = [
        Message(role="user", content="hello"),
    ]
    with pytest.raises(
        ValueError, match="Messages must be at least length 2, but got 1 messages"
    ):
        validate_messages(messages)

    # Test repeated user message
    messages = [
        Message(role="user", content="hello"),
        Message(role="user", content="world"),
        Message(role="assistant", content="world"),
    ]
    with pytest.raises(
        ValueError, match="Two consecutive user messages at index 1 and 0 in messages"
    ):
        validate_messages(messages)

    # Test assistant message comes first
    messages = [
        Message(role="assistant", content="hello"),
        Message(role="user", content="world"),
    ]
    with pytest.raises(
        ValueError,
        match="Assistant message before expected user message at index 0 in messages",
    ):
        validate_messages(messages)


def test_split_text_by_image_tag():
    # Test single image tag in the middle
    text = "hello <image>world"
    assert split_text_by_image_tag(text, "<image>") == [
        {"type": "text", "content": "hello "},
        {"type": "image"},
        {"type": "text", "content": "world"},
    ]

    # Test multiple image tags and image tag in beginning
    text = "[image]hello [image]world"
    assert split_text_by_image_tag(text, "[image]") == [
        {"type": "image"},
        {"type": "text", "content": "hello "},
        {"type": "image"},
        {"type": "text", "content": "world"},
    ]

    # Test an image tag that is not present in the text
    text = "hello world"
    assert split_text_by_image_tag(text, "asdfghjkl;") == [
        {"type": "text", "content": "hello world"}
    ]

    # Test consecutive image tags
    text = "<image><image>hello <image>world"
    assert split_text_by_image_tag(text, "<image>") == [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "content": "hello "},
        {"type": "image"},
        {"type": "text", "content": "world"},
    ]

    # Test image tag at the end
    text = "hello <image>"
    assert split_text_by_image_tag(text, "<image>") == [
        {"type": "text", "content": "hello "},
        {"type": "image"},
    ]
