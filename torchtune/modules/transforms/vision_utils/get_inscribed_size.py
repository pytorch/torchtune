# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math

from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_inscribed_size(
    image_size: Tuple[int], target_size: Tuple[int], max_size: Optional[int]
) -> Tuple[int]:
    """
    Calculates the size of an image, if it was resized to be inscribed within the target_size.
    It is upscaled or downscaled such that one size is equal to the target_size, and the second
    size is less than or equal to the target_size.

    The user can set max_size to limit upscaling along the larger dimension when target_size exceeds
    the image_size.

    Args:
        image_size (Tuple[int]): The size of the image, in the form [height, width].
        target_size (Tuple[int]): The desired resolution to fit the image into, in the format [height, width].
        max_size (Optional[int]): The maximum size to upscale the image to.
            If None, will upscale to target size.

    Returns:
        Tuple[int]: The resize dimensions for the image, in the format [height, width].

    Examples:

        Example 1: The image will be upscaled from (300, 800) to (448, 1194), since 448 is the limiting side.

            >>> max_size = None
            >>> image_size = (300, 800)
            >>> target_size = (448, 1344)
            >>> output = get_inscribed_size(image_size, target_size, max_size)

        Example 2: The image will stay as is, since 800 > 600.

            >>> max_size = 600
            >>> image_size = (300, 800)
            >>> target_size = (448, 1344)
            >>> output = get_inscribed_size(image_size, target_size, max_size)

        Example 3: The image will be downscaled from (500, 1000) to (224, 448).

            >>> max_size = 600
            >>> image_size = (500, 1000)
            >>> target_size = (448, 488)
            >>> output = get_inscribed_size(image_size, target_size, max_size)
    """
    assert len(image_size) == 2, "Image size must be a list of length 2."
    assert len(target_size) == 2, "Canvas size must be a list of length 2."

    image_height, image_width = image_size

    # Bound target_size with max_size.
    if max_size is not None:
        target_height = min(max(image_height, max_size), target_size[0])
        target_width = min(max(image_width, max_size), target_size[1])
    else:
        target_height, target_width = target_size

    # Calculate the largest aspect ratio preserving size that fits target_size.
    scale_h = target_height / image_height
    scale_w = target_width / image_width

    resize_height = min(math.floor(image_height * scale_w), target_height)
    resize_width = min(math.floor(image_width * scale_h), target_width)

    return (resize_height, resize_width)
