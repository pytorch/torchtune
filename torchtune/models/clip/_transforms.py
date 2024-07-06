# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, List, Mapping, Optional, Tuple

import torch
import torchvision
from PIL import Image

from torchtune.modules.transforms import (
    find_supported_resolutions,
    get_canvas_best_fit,
    resize_with_pad,
    tile_crop,
)

from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)


class CLIPImageTransform:
    """
    This class accepts images of any size and dynamically resizes, pads, normalizes and tiles it
    based on the image aspect ratio and the number of image tiles we allow.

    The algorithm will NOT distort the image to fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    The user can choose if they want to allow upscaling by using the flag ``resize_to_max_canvas``.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image tiles, with side 224px, then:

    If ``resize_to_max_canvas=False``, then:
    best_resolution = (448, 896) -> smallest canvas, up to 16 tiles, that doesn't require downscaling
    image is NOT resized
    image is padded (300, 800) -> 448,896
    Image is tiled 2x4, for a final output shape of (8, 3, 224, 224)

    If ``resize_to_max_canvas=True``, then:
    best_resolution = (448, 1344) # canvas that allows maximum upscaling, with minimum padding, up to 16 tiles
    image is resized without distortion (300,800) -> (448, 1194) #448 is the limiting side for the resize
    image is padded (448, 1194) -> (448, 1344)
    Image is tiled 2x5, for a final output shape of (10, 3, 224, 224)

    Args:
        image_mean (Optional[List[float]]): Mean values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed.
        image_std Union[float, List[float]]]): Standard deviation values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed.
        possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
            where each tuple represents a possible canvas to fit the image into when calling ``get_canvas_best_fit``.
            If None, this will be calculated using max_num_tiles and tile_size.
        tile_size (int): Size of the tiles to divide the image into
        max_num_tiles (Optional[int]): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
        resample (str): Resampling method used when resizing images. Supports any enum of
            ``torchvision.transforms.InterpolationMode``, e.g. "nearest", "nearest_exact", "bilinear", "bicubic".
        resize_to_max_canvas (bool):
            "If True, the image will be upscaled without distortion to fit the largest possible
            resolution from possible_resolutions.
            If False, it will pick the resolution that minimizes downscaling, including no downscaling at all.
            In this case, the image will only be upscaled if it's size < tile_size.

    Examples:
        >>> image_transform = CLIPImageTransform(
        ...    image_mean=None,
        ...    image_std=None,
        ...    tile_size=224,
        ...    possible_resolutions=None,
        ...    max_num_tiles=4,
        ...    resample="bilinear",
        ...    resize_to_max_canvas=True,
        ...)
        >>> # create random image
        >>> image = (np.random.rand(100,200,3) * 255).astype(np.uint8)
        >>> image = PIL.Image.fromarray(image)
        >>> output = image_transform(image)
        >>> output['image'].shape # [num_tiles, num_channels, tile_size, tile_size]
        torch.Size([2, 3, 224, 224])
        >>> output['ar'] # image best fits the canvas 224x448
        torch.tensor([1,2])
    """

    def __init__(
        self,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        possible_resolutions: Optional[List[Tuple[int, int]]] = None,
        tile_size: int = 224,
        max_num_tiles: Optional[int] = 4,
        resample: str = "bilinear",
        resize_to_max_canvas: bool = False,
    ) -> None:

        # get_canvas_best_fit
        assert (
            possible_resolutions is not None or max_num_tiles is not None
        ), f"Either possible_resolutions or max_num_tiles must be given. Got {possible_resolutions=} and {max_num_tiles=}"

        # If possible_resolutions are not given, then calculate possible ones based on max_num_tiles
        if not possible_resolutions and max_num_tiles:
            possible_resolutions = find_supported_resolutions(
                max_num_tiles=max_num_tiles, tile_size=tile_size
            )
        else:
            possible_resolutions = possible_resolutions

        self.possible_resolutions = torch.tensor(possible_resolutions).reshape(-1, 2)
        logger.info(
            f"Found possible_resolutions: {self.possible_resolutions}. Will fit the images into the canvas with best fit."
        )

        self.resize_to_max_canvas = resize_to_max_canvas

        # normalize
        assert (image_mean is None) == (
            image_std is None
        ), f"Need to provide both or none of image_mean and image_std. Got {image_mean=} and {image_std=}"
        self.image_mean = image_mean
        self.image_std = image_std

        # resize_with_pad
        self.max_upscaling_size = None if resize_to_max_canvas else tile_size
        self.resample = torchvision.transforms.InterpolationMode[resample.upper()]

        # tile_crop
        self.tile_size = tile_size

    def __call__(self, *, image: Image.Image, **kwargs) -> Mapping[str, Any]:

        assert isinstance(image, Image.Image), "Input image must be a PIL image."

        # Make image torch.tensor((3, H, W), dtype='float32'), 0<=values<=1
        image_tensor = F.to_dtype(
            F.grayscale_to_rgb_image(F.to_image(image)), scale=True
        )

        # Find the best canvas to fit the image without distortion
        best_resolution = get_canvas_best_fit(
            image=image_tensor,
            possible_resolutions=self.possible_resolutions,
            resize_to_max_canvas=self.resize_to_max_canvas,
        )

        # resize without distortion + pad to fit best_resolution
        image_tensor = resize_with_pad(
            image=image_tensor,
            target_size=best_resolution,
            resample=self.resample,
            max_upscaling_size=self.max_upscaling_size,
        )

        # Normalize
        if self.image_mean and self.image_std:
            image_tensor = F.normalize(
                image_tensor, mean=self.image_mean, std=self.image_std
            )

        # Divide the image into equally sized tiles
        image_tensor = tile_crop(image=image_tensor, tile_size=self.tile_size)

        aspect_ratio = torch.tensor(best_resolution).reshape(-1) // self.tile_size
        return {
            "image": image_tensor,
            "aspect_ratio": aspect_ratio,
        }
