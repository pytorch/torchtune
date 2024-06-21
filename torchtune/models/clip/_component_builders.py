# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING, List

if TYPE_CHECKING:
    from PIL import Image

import torch

from torchtune.modules.transforms.vision import (
    tile_crop,
    resize_with_pad,
    get_canvas_best_fit
)
from torchtune.modules.transforms.vision.get_canvas_best_fit import _find_supported_resolutions

from torchvision.transforms.v2 import functional as F
import torchvision

logger = logging.getLogger(__name__)

class CLIPImageTransform:
    """
    This class accepts images of any size and dynamically resize, pads and tiles it
    based on the image aspect ratio and the number of image tiles we allow.

    The algorithm will NOT distort the image to fit a certain aspect ratio, because
    that leads to a significant degradation in image quality.

    For example, if an input image is of size 300x800, and we want to allow
    a maximum of 16 image tiles, it will find the closest aspect ratio that
    is allowed within 16 image tiles, with some restrictions,
    ie.g., 2:4 = 2 horizontal tiles and 4 vertical tiles, giving a total of 8 tiles.

    The image will then be resized to products of the tile_size (default is
    224px), so in this case it will  be resized to 388:896 (without distortion) and padded to 448:896,
    where we maintain the original aspect ratio and pad with zeros value for the rest.
    This approach minimizes the amount of padding required for any arbitrary resolution.

    However, if limit_upscaling_to_tile_size is set to True, the upscaling will be limited to the tile size.
    In the example above, the image would remain 300x800 (no upscaling), and then padded to 448:896.

    The final output will therefore be of shape (8, 3, 224, 224), where 2x4
    tiles are coming from the resizing and tileing.

    Args:
        image_mean Union[float, List[float]]: Mean values for normalization.
            Should be the same used for the pre-trained model.
        image_std Union[float, List[float]]]): Standard deviation values for normalization.
            Should be the same used for the pre-trained model.
        tile_size (int): Size of the tiles to divide the image into.
        possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
        max_num_tiles (Optional[int]): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions, e.g. [[224, 224]] if max_num_tiles = 1 and tile_size = 224.
        resample (str): Resampling method used when resizing images. Supports "nearest", "nearest_exact", "bilinear", "bicubic"
        do_rescale (bool): Flag to determine whether to rescale the image by "rescale_factor".
        rescale_factor (Union[int, float]): Scale factor used if rescaling the image.
        do_normalize (bool): Flag to determine whether to normalize the image.
        limit_upscaling_to_tile_size (bool): If True, when target_resolution is larger than the image,
            the image will be upscaled to at most the tile_size before padding to target_resolution.
            This is **only** for upscaling and has no effect if the image is already larger than the tile size, or has to be
            downscaled.
    """

    def __init__(
        self,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        tile_size: int = 224,
        possible_resolutions: Optional[List[Tuple[int, int]]] = None,
        max_num_tiles: Optional[int] = 4,
        resample: str = "bilinear",
        limit_upscaling_to_tile_size: bool = True,
    ) -> None:

        # get_canvas_best_fit
        assert possible_resolutions is not None or max_num_tiles is not None, (
            f"Either possible_resolutions or max_num_tiles must be given. Got {possible_resolutions=} and {max_num_tiles=}")

        # If possible_resolutions are not given, then calculate possible ones based on max_num_tiles
        if not possible_resolutions and max_num_tiles:
            possible_resolutions = _find_supported_resolutions(
                max_num_tiles=max_num_tiles, tile_size=tile_size
            )
        else:
            possible_resolutions = possible_resolutions
        self.possible_resolutions = torch.tensor(possible_resolutions).reshape(-1, 2)
        logger.info(f"possible_resolutions: {self.possible_resolutions}")

        # normalize
        assert (
            (image_mean is None) == (image_std is None)
        ), f"Need to provide both or none of image_mean and image_std. Got {image_mean=} and {image_std=}"
        self.image_mean = image_mean
        self.image_std = image_std

        # resize_with_pad
        self.max_upscaling_size = tile_size if limit_upscaling_to_tile_size else None
        self.resample = torchvision.transforms.InterpolationMode[resample.upper()]  # e.g. "BILINEAR"

        # tile_crop
        self.tile_size = tile_size

    def __call__(
        self, 
        image: "Image.Image",
    ) -> Dict[str, torch.Tensor]:

        assert isinstance(image, "Image.Image"), "Input image must be a PIL image."

        # Make image have dimension [3, H, W]. Input can be grayscale, RGB, channels-first or last.
        image_tensor = F.to_dtype(F.grayscale_to_rgb_image(F.to_image(image)), scale=True)

        # Find the best canvas to fit the image without distortion
        best_resolution = get_canvas_best_fit(image=image_tensor, possible_resolutions=self.possible_resolutions)

        # resize without distortion + pad to fit best_resolution
        image_tensor = resize_with_pad(
            image=image_tensor, 
            target_size=best_resolution,
            resample=self.resample,
            max_upscaling_size=self.max_upscaling_size
        )

        # Normalize
        if self.image_mean and self.image_std:
            image_tensor = F.normalize(image_tensor, mean=self.image_mean, std=self.image_std)

        # Divide the image into equally sized tiles
        image_tensor = tile_crop(image=image_tensor, tile_size=self.tile_size)

        return {
            "image": image_tensor,
            "image_size": torch.tensor(best_resolution).reshape(-1),
        }
