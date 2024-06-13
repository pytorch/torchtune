# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Tuple, Union

import torch

import torchvision

from torchtune.modules.transforms.utils import get_max_res_without_distortion
from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)


def rescale(image: torch.tensor, scale: Union[float, int]) -> torch.tensor:
    """
    Rescale a torch tensor by scale amount.

    Args:
        image (torch.tensor): The image to rescale.
        scale (Union[float, int]): The scale to rescale the image by.

    Returns:
        torch.tensor: The rescaled image.
    """
    return image * scale


def divide_to_equal_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Divides an image into equally sized patches. Image size must be a multiplier of patch_size.

    Args:
        image (torch.Tensor): Image tensor.
        patch_size (int): Size of each patch.
    Returns:
        torch.Tensor: Tensor of shape [num_patches, channel_size, patch_size, patch_size]
    Example:
        >>> image = torch.rand(3, 200, 300)
        >>> patches = _divide_to_patches(image, patch_size=50)
        >>> patches.shape # 4x6
        torch.Size([24, 3, 50, 50])

        >>> image = torch.rand(3, 400, 600)
        >>> patches = _divide_to_patches(image, patch_size=200)
        >>> len(patches) # 2x3
        torch.Size([6, 3, 200, 200])
    """

    channel_size, height, width = image.shape

    # Reshape to split height and width into patch_size blocks
    patches_height = height // patch_size
    patches_width = width // patch_size

    reshaped = image.view(
        channel_size, patches_height, patch_size, patches_width, patch_size
    )

    # Transpose to bring patches together
    # We want [patches_height, patches_width, channel_size, patch_size, patch_size]
    transposed = reshaped.permute(1, 3, 0, 2, 4)

    # Flatten the patches
    patches = transposed.contiguous().view(
        patches_height * patches_width, channel_size, patch_size, patch_size
    )
    return patches


class ResizeWithoutDistortion:
    """
    Used to resize an image to target_resolution, without distortion.

    If target_size requires upscaling the image, the user can set max_upscaling_size to
    limit the upscaling to a maximum size. In this case, since we rescale without distortion,
    modifying target_size works as a boundary for the image's largest side.

    Ex 1: target_size = (1000, 1200), max_upscaling_size = 600, image_size = (400,200) --> target_size = (600, 600)
    >> new_size_without_distortion = (600, 300)
    Ex 2: target_size = (1000, 1200), max_upscaling_size = 600, image_size = (2000,200) --> target_size = (1000, 600)
    >> new_size_without_distortion = (1000, 100)
    Ex 3: target_size = (1000, 1200), max_upscaling_size = 2000, image_size = (400,200) --> target_size = (1000, 1200)
    >> new_size_without_distortion = (1000, 500)
    Ex 4: target_size = (1000, 1200), max_upscaling_size = None, image_size = (400,200) --> target_size = (1000, 1200)
    >> new_size_without_distortion = (1000, 500)

    Args:
        resample (str): Resampling method used when resizing images. Supports "nearest", "nearest_exact", "bilinear", "bicubic".
        max_upscaling_size (int): The maximum size to upscale the image to.
            If None, there is no limit.
    """

    def __init__(self, resample: str, max_upscaling_size: Optional[int] = None):
        self.resample = torchvision.transforms.InterpolationMode[resample.upper()]
        self.max_upscaling_size = max_upscaling_size

    def __call__(
        self, image: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:

        image_size = image.shape[-2:]

        # If target_size requires upscaling, we might want to limit the upscaling to self.max_upscaling_size
        if self.max_upscaling_size is not None:
            new_target_height = min(
                max(image_size[0], self.max_upscaling_size), target_size[0]
            )
            new_target_width = min(
                max(image_size[1], self.max_upscaling_size), target_size[1]
            )
            target_size = (new_target_height, new_target_width)

        # resize to target_size while preserving aspect ratio
        new_size_without_distortion = get_max_res_without_distortion(
            image_size, target_size
        )

        image = F.resize(
            image, new_size_without_distortion, interpolation=self.resample
        )

        return image


def pad_image_top_left(
    image: torch.Tensor, target_size: Tuple[int, int]
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

    padding = (0, 0, pad_x, pad_y)
    return F.pad(inpt=image, padding=padding)
