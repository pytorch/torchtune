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

from torchtune.modules.transforms.vision_utils.get_canvas_best_fit import (
    find_supported_resolutions,
    get_canvas_best_fit,
)
from torchtune.modules.transforms.vision_utils.get_inscribed_size import (
    get_inscribed_size,
)
from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop

from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)


class _CLIPImageTransform(torch.nn.Module):
    def __init__(
        self,
        resample: str,
        image_mean: Optional[List[float]],
        image_std: Optional[List[float]],
        tile_size: int,
        max_num_tiles: int,
        antialias: bool,
    ):
        super().__init__()
        self.resample = resample
        self.image_mean = image_mean
        self.image_std = image_std
        self.tile_size = tile_size
        self.max_num_tiles = max_num_tiles
        self.antialias = antialias
        self.tile_crop = tile_crop
        self.pad = torch.nn.functional.pad

    def check_variable_bounds_for_export(
        self,
        target_size: List[int],
        canvas_size: List[int],
        lower: int,
        upper: int,
    ) -> None:
        """
        Performs torch._checks used to export the model. For eager mode usage, please disregard.
        The check mitigates data dependent errors that may occur during torch.export. It installs a
        deferred runtime assert, instead of a compile-time guard. Data dependent errors usually occur
        in models with data-dependent control flow, eg. via .item(), tolist(), nonzero(). For more
        context: https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit
        """
        # Check lower <= canvas_size <= upper.
        for var in canvas_size:
            torch._check(var >= lower)
            torch._check(var <= upper)

        # Check lower <= target_size <= canvas_size.
        for i in range(len(target_size)):
            torch._check(target_size[i] >= lower)
            torch._check(target_size[i] <= canvas_size[i])

    def forward(
        self, image: torch.Tensor, target_size: torch.Tensor, canvas_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the core transformations involved in CLIPImageTransform;
        1. Resize the image to target_size.
        2. Pad the image to canvas_size.
        3. Normalize the image using image_mean and image_std.
        4. Reshape the image tensor into [n, channels, tile_size, tile_size].
        Args:
            image (torch.Tensor): image as a 3D tensor in form [C, H, W].
            target_size (torch.Tensor): tensor of shape (2,) containing the target_height and target_width for resize.
            canvas_size (torch.Tensor): tensor of shape (2,) containing the canvas_height and canvas_width for padding.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor of shape [n, channels, tile_size, tile_size]
                and aspect ratio tensor of shape [1, 2].
        """

        target_h, target_w = target_size.tolist()
        canvas_h, canvas_w = canvas_size.tolist()

        # Checks to allow the model to export via torch.export.
        self.check_variable_bounds_for_export(
            target_size=[target_h, target_w],
            canvas_size=[canvas_h, canvas_w],
            lower=2,
            upper=self.tile_size * self.max_num_tiles,
        )

        # Resize.
        image = torchvision.transforms._functional_tensor.resize(
            image,
            size=[target_h, target_w],
            interpolation=self.resample,
            antialias=self.antialias,
        )

        # Pad, such that the image is on the top-left and padded on the right-bottom.
        # padding = [left, right, top, bottom]
        padding = [0, canvas_w - target_w, 0, canvas_h - target_h]
        output = self.pad(image, padding)

        # Normalize.
        if self.image_mean is not None and self.image_std is not None:
            output = F.normalize(output, self.image_mean, self.image_std)

        # Reshape.
        tiles = self.tile_crop(output, self.tile_size)

        # Calculate aspect ratio.
        aspect_ratio = canvas_size // self.tile_size

        return tiles, aspect_ratio


class CLIPImageTransform:
    """
    Note: this class is functionally the same as CLIPImageTransform from torchtune/models/clip/_transforms
    and should produce identical output results. This version is structured to be (more easily)
    exported via torch.export for inference use cases in, eg. ExecuTorch or AOTI.

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
            Should be the same used for the pre-trained model. If None, no normalization is performed. Default None.
        image_std (Optional[List[float]]): Standard deviation values of each channel, used for normalization.
            Should be the same used for the pre-trained model. If None, no normalization is performed. Default None.
        possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
            where each tuple represents a possible canvas to fit the image into when calling ``get_canvas_best_fit``.
            If None, this will be calculated using max_num_tiles and tile_size. Default None.
        tile_size (int): Size of the tiles to divide the image into. Default 224.
        max_num_tiles (Optional[int]): Only used if possible_resolutions is NOT given.
            Maximum number of tiles to break an image into.
            This will be used to generate possible_resolutions,
            e.g. [(224, 224), (224, 448), (448, 224)] if max_num_tiles = 2 and tile_size = 224.
            Default 4.
        resample (str): Resampling method used when resizing images. Supports any enum of
            ``torchvision.transforms.InterpolationMode``, e.g. "nearest", "nearest_exact", "bilinear", "bicubic".
            Default 'bilinear'.
        resize_to_max_canvas (bool): "If True, the image will be upscaled without distortion to fit the largest possible
            resolution from possible_resolutions.
            If False, it will pick the resolution that minimizes downscaling, including no downscaling at all.
            In this case, the image will only be upscaled if it's size < tile_size. Default False.
        antialias (bool): Whether to apply antialiasing when resizing the image. Default True.
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
        antialias: bool = True,
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
        self.max_num_tiles = max_num_tiles

        # normalize
        assert (image_mean is None) == (
            image_std is None
        ), f"Need to provide both or none of image_mean and image_std. Got {image_mean=} and {image_std=}"
        self.image_mean = image_mean
        self.image_std = image_std

        # resize
        self.max_size = None if resize_to_max_canvas else tile_size
        self.resample = resample
        self.antialias = antialias

        # tile_crop
        self.tile_size = tile_size

        self.core_transform = _CLIPImageTransform(
            resample=self.resample,
            image_mean=self.image_mean,
            image_std=self.image_std,
            tile_size=self.tile_size,
            max_num_tiles=self.max_num_tiles,
            antialias=self.antialias,
        )

    def __call__(self, *, image: Image.Image, **kwargs) -> Mapping[str, Any]:

        assert isinstance(image, Image.Image), "Input image must be a PIL image."

        # Make image torch.tensor((3, H, W), dtype='float32'), 0<=values<=1.
        image_tensor = F.to_dtype(
            F.grayscale_to_rgb_image(F.to_image(image)), scale=True
        )

        # Find the best canvas to fit the image without distortion.
        best_resolution = get_canvas_best_fit(
            image=image_tensor,
            possible_resolutions=self.possible_resolutions,
            resize_to_max_canvas=self.resize_to_max_canvas,
        )

        # Find the dimensions of the image, such that it is inscribed within best_resolution.
        inscribed_size = get_inscribed_size(
            image_tensor.shape[-2:], best_resolution, self.max_size
        )

        # Call _CLIPImageTransform to perform resize, pad, normalize and reshape transforms.
        tiles, aspect_ratio = self.core_transform(
            image=image_tensor,
            target_size=torch.tensor(inscribed_size),
            canvas_size=torch.tensor(best_resolution),
        )

        kwargs.update(
            {
                "image": tiles,
                "aspect_ratio": aspect_ratio,
            }
        )

        return kwargs
