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
    limit the upscaling to a maximum size.

    Example 1:
    max_upscaling_size = None
    image = torch.rand([3, 300, 800])
    target_size = (448, 1344)

    The image will be upscaled (300,800) -> (448, 1194), since 448 is the limiting side,
    and then padded (448, 1194) -> (448, 1344)

    Example 2:
    max_upscaling_size = 600
    image = torch.rand([3, 300, 800])
    target_size = (448, 1344)

    The image will NOT be upscaled, since 800>600,
    and then padded (300, 800) -> (448, 1344)

    Args:
        image (torch.Tensor): The input image tensor in the format [..., H, W].
        target_size (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].
        resample (str): Resampling method used when resizing images.
            Supports torchvision.transforms.InterpolationMode.NEAREST,
            NEAREST_EXACT, BILINEAR, BICUBIC
        max_upscaling_size (int): The maximum size to upscale the image to.
            If None, will upscale up to target_size.
    """

    image_height, image_width = image.shape[-2:]
    image_size = (image_height, image_width)

    # If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
    if max_upscaling_size is not None:
        new_target_height = min(max(image_height, max_upscaling_size), target_size[0])
        new_target_width = min(max(image_width, max_upscaling_size), target_size[1])
        target_size_resize = (new_target_height, new_target_width)
    else:
        target_size_resize = target_size

    # resize to target_size while preserving aspect ratio
    new_size_without_distortion = _get_max_res_without_distortion(
        image_size=image_size,
        target_size=target_size_resize,
    )

    image = F.resize(
        inpt=image,
        size=list(new_size_without_distortion),
        interpolation=resample,
        antialias=True,
    )

    image = _pad_image_top_left(image=image, target_size=target_size)

    return image


def _pad_image_top_left(
    image: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Places the image at the top left of the canvas and pads with 0 the right and bottom
    to fit to the target resolution. If target_size < image_size, it will crop the image.

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

    For example, if image_size = (200,400) and target_size = (600,800),
    scale_h = 600/200 = 3
    scale_w = 800/400 = 2
    So the maximum that we can upscale without distortion is min(scale_h, scale_w) = 2

    Since scale_w is the limiting side, then new_w = target_w, and new_h = old_h*scale_w

    Args:
        image_size (Tuple[int, int]): The original resolution of the image.
        target_resolution (Tuple[int, int]): The desired resolution to fit the image into.
    Returns:
        Tuple[int, int]: The optimal dimensions to which the image should be resized.
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
