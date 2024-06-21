# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
from typing import Optional, Tuple

import torch

import torchvision
from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)


def resize_with_pad(
    image: torch.Tensor,
    target_size: Tuple[int, int],
    resample: torchvision.transforms.InterpolationMode,
    max_upscaling_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Used to resize+pad an image to target_resolution, without distortion.

    If target_size requires upscaling the image, the user can set max_upscaling_size to
    limit the upscaling to a maximum size, which overwrites target_size if it is smaller.

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].
        resample (str): Resampling method used when resizing images.
            Supports torchvision.transforms.InterpolationMode.NEAREST,
            NEAREST_EXACT, BILINEAR, BICUBIC
        max_upscaling_size (int): The maximum size to upscale the image to.
            If None, will upscale up to target_size.
    Examples:
    >>> target_size = (1000, 1200)
    >>> max_upscaling_size = 600
    >>> image_size = (400, 200)
    >>> ResizeWithoutDistortion(max_upscaling_size=max_upscaling_size)(image_size, target_size)
    (600, 300)  # new_size_without_distortion

    >>> target_size = (1000, 1200)
    >>> max_upscaling_size = 600
    >>> image_size = (2000, 200)
    >>> ResizeWithoutDistortion(max_upscaling_size=max_upscaling_size)(image_size, target_size)
    (1000, 100)  # new_size_without_distortion

    >>> target_size = (1000, 1200)
    >>> max_upscaling_size = 2000
    >>> image_size = (400, 200)
    >>> ResizeWithoutDistortion(max_upscaling_size=max_upscaling_size)(image_size, target_size)
    (1000, 500)  # new_size_without_distortion

    >>> target_size = (1000, 1200)
    >>> max_upscaling_size = None
    >>> image_size = (400, 200)
    >>> ResizeWithoutDistortion(max_upscaling_size=max_upscaling_size)(image_size, target_size)
    (1000, 500)  # new_size_without_distortion
    """

    image_height, image_width = image.shape[-2:]
    image_size = (image_height, image_width)

    # If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
    if max_upscaling_size is not None:
        new_target_height = min(max(image_height, max_upscaling_size), target_size[0])
        new_target_width = min(max(image_width, max_upscaling_size), target_size[1])
        target_size = (new_target_height, new_target_width)

    # resize to target_size while preserving aspect ratio
    new_size_without_distortion = _get_max_res_without_distortion(
        image_size=image_size,
        target_size=target_size,
    )

    image = F.resize(
        inpt=image, size=list(new_size_without_distortion), interpolation=resample
    )

    image = _pad_image_top_left(image=image, target_size=target_size)

    # assert shapes
    _, height, width = image.shape
    assert (
        height == target_size[0] and width == target_size[1]
    ), f"Expected image shape {target_size} but got {height}x{width}"

    return image


def _pad_image_top_left(
    image: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Places the image at the top left of the canvas and pads the right and bottom
    to fit to the target resolution. If padding is negative, the image will be cropped.

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].

    Returns:
        torch.Tensor: The padded image tensor in the format [..., H, W].

    """

    image_size = image.shape[-2:]

    height, width = image_size
    target_height, target_width = target_size

    pad_x = target_width - width
    pad_y = target_height - height

    padding = [0, 0, pad_x, pad_y]
    return F.pad(inpt=image, padding=padding)


def _get_max_res_without_distortion(
    image_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[int, int]:

    """
    Determines the maximum resolution to which an image can be resized to without distorting its
    aspect ratio, based on the target resolution.

    Args:
        image_size (Tuple[int, int]): The original resolution of the image (height, width).
        target_resolution (Tuple[int, int]): The desired resolution to fit the image into (height, width).
    Returns:
        Tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.
    Example:
        >>> _get_max_res_without_distortion([200, 300], target_size = [450, 200])
        (134, 200)
        >>> _get_max_res_without_distortion([800, 600], target_size = [450, 1300])
        (450, 338)
    """

    original_height, original_width = image_size
    target_height, target_width = target_size

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.floor(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.floor(original_width * scale_h), target_width)

    return new_height, new_width
