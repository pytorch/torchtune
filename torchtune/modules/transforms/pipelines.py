# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import PIL

import torch

from torchtune.modules.transforms.transforms import (
    divide_to_equal_patches,
    pad_image_top_left,
    rescale,
    ResizeWithoutDistortion,
)
from torchtune.modules.transforms.utils import GetBestResolution
from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)

ImageInput = Union["PIL.Image.Image", np.ndarray, "torch.Tensor"]


class VariableImageSizeTransforms:
    """
    This class accepts images of any size and dynamically resize, pads and chunks it
    based on the image aspect ratio and the number of image chunks we allow.

    The algorithm will NOT distort the image fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image chunks, it will find the closest aspect ratio that
    is allowed within 16 image chunks, with some restrictions,
    ie.g., 2:4 = 2 horizontal patches and 4 vertical patches, giving a total of 8 chunks.

    The image will then be resized to products of the patch_size (default is
    224px), so in this case it will  be resized to 388:896 (without distortion) and padded to 448:896,
    where we maintain the original aspect ratio and pad with zeros value for the rest.
    This approach minimizes the amount of padding required for any arbitrary resolution.

    However, if limit_upscaling_to_patch_size is set to True, the upscaling will be limited to the patch size.
    In the example above, the image would remain 300x800 (no upscaling), and then padded to 448:896.

    The final output will therefore be of shape (8, 3, 224, 224), where 2x4
    patches are coming from the resizing and chunking.

    Args:
        image_mean Union[float, List[float]]: Mean values for normalization.
            Should be the same used for the pre-trained model.
        image_std Union[float, List[float]]]): Standard deviation values for normalization.
            Should be the same used for the pre-trained model.
        patch_size (int): Size of the patches to divide the image into.
        possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
        max_num_chunks (Optional[int]): Only used if possible_resolutions is NOT given.
            Maximum number of chunks to break an image into.
            This will be used to generate possible_resolutions, e.g. [[224, 224]] if max_num_chunks = 1 and patch_size = 224.
        resample (str): Resampling method used when resizing images. Supports "nearest", "nearest_exact", "bilinear", "bicubic"
        do_rescale (bool): Flag to determine whether to rescale the image by "rescale_factor".
        rescale_factor (Union[int, float]): Scale factor used if rescaling the image.
        do_normalize (bool): Flag to determine whether to normalize the image.
        limit_upscaling_to_patch_size (bool): If True, when target_resolution is larger than the image,
            the image will be upscaled to at most the patch_size before padding to target_resolution.
            This is **only** for upscaling and has no effect if the image is already larger than the patch size, or has to be
            downscaled.
    """

    def __init__(
        self,
        image_mean: Optional[Union[float, Tuple[float, float, float]]],
        image_std: Optional[Union[float, Tuple[float, float, float]]],
        patch_size: int = 224,
        possible_resolutions: Optional[Tuple[Tuple[int, int]]] = None,
        max_num_chunks: Optional[int] = 4,
        resample: str = "bilinear",
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        limit_upscaling_to_patch_size: bool = True,
    ) -> None:

        logger.info("Initializating ImageProcessor...")

        assert (
            possible_resolutions or max_num_chunks
        ), "Either possible_resolutions or max_num_chunks must be given."

        self.patch_size = patch_size

        # normalize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

        # rescale
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor

        # best resolution
        self.get_best_resolution = GetBestResolution(
            possible_resolutions=possible_resolutions,
            max_num_chunks=max_num_chunks,
            patch_size=patch_size,
        )

        # resizing
        max_upscaling_size = self.patch_size if limit_upscaling_to_patch_size else None
        self.resize_without_distortion = ResizeWithoutDistortion(
            resample=resample, max_upscaling_size=max_upscaling_size
        )

    def __call__(self, image: ImageInput) -> Dict[str, torch.Tensor | Tuple[int, int]]:

        # Make image have dimension [3, H, W]. Input can be grayscale, RGB, channels-first or last.
        image = F.grayscale_to_rgb_image(F.to_image(image))
        _, height, width = image.shape
        image_size = (height, width)

        # Find the best canvas from possible_resolutions to fit the image without distortion
        best_resolution = self.get_best_resolution(image_size)

        # resize to best_resolution while preserving aspect ratio
        image = self.resize_without_distortion(image=image, target_size=best_resolution)

        # pad to fit the best resolution
        image = pad_image_top_left(image=image, target_size=best_resolution)

        # Normalize and rescale
        image = image.float()
        if self.do_rescale:
            image = rescale(image, self.rescale_factor)

        if self.do_normalize:
            image = F.normalize(image, mean=self.image_mean, std=self.image_std)

        # Divide the image into equally squared patches
        image = divide_to_equal_patches(image, patch_size=self.patch_size)

        return {"pixel_values": image, "image_size": (height, width)}
