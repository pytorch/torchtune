# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
from PIL import Image

from tests.common import ASSETS
from torchtune.data import (
    format_content_with_images,
    Message,
    PromptTemplate,
    truncate,
    validate_messages,
)
from torchtune.data._utils import _get_prompt_template, load_image
from torchtune.models.llama2 import Llama2ChatTemplate


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


def test_format_content_with_images():
    test_image_1 = Image.new(mode="RGB", size=(4, 4))
    test_image_2 = Image.new(mode="RGB", size=(4, 4))
    test_image_3 = Image.new(mode="RGB", size=(4, 4))

    # Test single image tag in the middle
    text = "hello <image>world"
    assert format_content_with_images(
        text,
        image_tag="<image>",
        images=[test_image_1],
    ) == [
        {"type": "text", "content": "hello "},
        {"type": "image", "content": test_image_1},
        {"type": "text", "content": "world"},
    ]

    # Test multiple image tags and image tag in beginning
    text = "[image]hello [image]world"
    assert format_content_with_images(
        text,
        image_tag="[image]",
        images=[test_image_1, test_image_2],
    ) == [
        {"type": "image", "content": test_image_1},
        {"type": "text", "content": "hello "},
        {"type": "image", "content": test_image_2},
        {"type": "text", "content": "world"},
    ]

    # Test an image tag that is not present in the text
    text = "hello world"
    assert format_content_with_images(text, image_tag="asdfghjkl;", images=[]) == [
        {"type": "text", "content": "hello world"}
    ]

    # Test consecutive image tags
    text = "<image><image>hello <image>world"
    assert format_content_with_images(
        text,
        image_tag="<image>",
        images=[test_image_1, test_image_2, test_image_3],
    ) == [
        {"type": "image", "content": test_image_1},
        {"type": "image", "content": test_image_2},
        {"type": "text", "content": "hello "},
        {"type": "image", "content": test_image_3},
        {"type": "text", "content": "world"},
    ]

    # Test image tag at the end
    text = "hello <image>"
    assert format_content_with_images(
        text,
        image_tag="<image>",
        images=[test_image_1],
    ) == [
        {"type": "text", "content": "hello "},
        {"type": "image", "content": test_image_1},
    ]

    # Test errors when the number of images does not match the number of image tags
    text = "hello <image>world"
    with pytest.raises(
        ValueError,
        match="does not match number of image tags",
    ):
        format_content_with_images(
            text, image_tag="<image>", images=[test_image_1, test_image_2]
        )


def test_load_image(monkeypatch, tmp_path):
    tmp_image = str(ASSETS / "dog_on_skateboard.jpg")

    # Test loading from local file
    image = load_image(tmp_image)
    assert isinstance(image, Image.Image)
    assert image.size == (580, 403)

    # Test loading from remote file
    # Mock the urlopen function to return a BytesIO object
    def mock_urlopen(url):
        return open(tmp_image, "rb")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    image = load_image("http://example.com/test_image.jpg")
    assert isinstance(image, Image.Image)
    assert image.size == (580, 403)

    # Test that a ValueError is raised when the image path is invalid
    with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
        load_image("invalid_path")

    # Test a temporary file with invalid image data
    image_path = tmp_path / "test_image.jpg"
    with open(image_path, "w") as f:
        f.write("Invalid image data")

    # Test that a ValueError is raised when the image data is invalid
    with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
        load_image(str(image_path))

    # Test that a ValueError is raised when there is an HTTP error
    # Mock the urlopen function to raise an exception
    def mock_urlopen(url):
        raise Exception("Failed to load image")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    with pytest.raises(ValueError, match="Failed to load image"):
        load_image("http://example.com/test_image.jpg")

    # Test that a ValueError is raised when there is an IO error
    # Create a temporary file that cannot be read
    image_path = tmp_path / "test_image.jpg"
    with open(image_path, "w") as f:
        f.write("Test data")
    os.chmod(image_path, 0o000)  # Remove read permissions
    with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
        load_image(str(image_path))
    os.chmod(image_path, 0o644)  # Restore read permissions

    # Test that a ValueError is raised with invalid image data is read
    # Create a temporary file with invalid image data
    image_path = tmp_path / "test_image.jpg"
    with open(image_path, "wb") as f:
        f.write(b"Invalid image data")
    with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
        load_image(str(image_path))


def test_get_prompt_template():
    template = _get_prompt_template("torchtune.models.llama2.Llama2ChatTemplate")
    assert isinstance(template, Llama2ChatTemplate)

    template = _get_prompt_template({"user": ("1", "2"), "assistant": ("3", "4")})
    assert isinstance(template, PromptTemplate)
    assert template.template["user"] == ("1", "2")
    assert template.template["assistant"] == ("3", "4")

    with pytest.raises(
        ValueError,
        match="Prompt template must be a dotpath string or dictionary with custom template",
    ):
        _ = _get_prompt_template(["user", "assistant"])
