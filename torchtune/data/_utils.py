# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union
from urllib import request

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


def load_image(image_loc: Union[Path, str]) -> "PIL.Image.Image":
    """
    Convenience method to load an image in PIL format from a local file path or remote source.

    Args:
        image_loc (Union[Path, str]): Local file path or remote source pointing to the image
            which will be loaded in PIL format.

    Note:
        If loading an image from a remote source, the function expects the URL provided in ``image_loc``
        to start with "http" or "https" e.g. "https://www.wikipedia.org/en/bird.jpg".

    Raises:
        ValueError: If the image cannot be loaded from remote source.
        ValueError: If the image cannot be opened as a :class:`~PIL.Image.Image`.

    Examples:
        >>> # Load from remote source
        >>> image = load_image("https://www.wikipedia.org/en/bird.jpg")

        >>> # Load from local file path
        >>> image = load_image(Path("/home/user/bird.jpg"))

    Returns:
        PIL.Image.Image: The loaded image.
    """
    # Hackily import PIL to avoid burdensome import in the main module
    # TODO: Fix this
    from PIL import Image

    # If pointing to remote source, try to load to local
    if isinstance(image_loc, str) and image_loc.startswith("http"):
        try:
            image_loc = request.urlopen(image_loc)
        except Exception as e:
            raise ValueError(f"Failed to load image from {image_loc}") from e

    # Open the local image as a PIL image
    try:
        image = Image.open(image_loc)
    except Exception as e:
        raise ValueError(f"Failed to open image as PIL Image from {image_loc}") from e

    return image


def format_content_with_images(
    content: str, *, image_tag: str, images: List["PIL.Image.Image"]
) -> List[Dict[str, Any]]:
    """
    Given a raw text string, split by the specified ``image_tag``
    and form into list of dictionaries to be used in the :class:`~torchtune.data.Message` content
    field::

        [
            {
                "role": "system" | "user" | "assistant",
                "content":
                    [
                        {"type": "image", "content": <PIL.Image.Image>},
                        {"type": "text", "content": "This is a sample image."},
                    ],
            },
            ...
        ]

    Args:
        content (str): raw message text
        image_tag (str): string to split the text by
        images (List["PIL.Image.Image"]): list of images to be used in the content

    Raises:
        ValueError: If the number of images does not match the number of image tags in the content

    Examples:
        >>> content = format_content_with_images(
        ...     "<|image|>hello <|image|>world",
        ...     image_tag="<|image|>",
        ...     images=[<PIL.Image.Image>, <PIL.Image.Image>]
        ... )
        >>> print(content)
        [
            {"type": "image", "content": <PIL.Image.Image>},
            {"type": "text", "content": "hello "},
            {"type": "image", "content": <PIL.Image.Image>},
            {"type": "text", "content": "world"}
        ]

    Returns:
        List[Dict[str, Any]]: list of dictionaries to be used in the :class:`~torchtune.data.Message` content field
    """
    num_image_tags_in_content = content.count(image_tag)
    if len(images) != num_image_tags_in_content:
        raise ValueError(
            f"Number of images ({len(images)}) does not match number of image tags "
            f"({num_image_tags_in_content}) in content: {content}"
        )

    split_content = content.split(image_tag)
    final_content_list = []
    for i, substr in enumerate(split_content):
        if len(substr) > 0:
            final_content_list.append({"type": "text", "content": substr})
        if i < len(split_content) - 1:
            final_content_list.append({"type": "image", "content": images.pop(0)})

    return final_content_list
