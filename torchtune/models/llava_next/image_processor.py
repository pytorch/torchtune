# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import PIL

import torch

import torchvision
from torch import nn

from torchtune.utils.image_transforms_utils import find_supported_resolutions
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

logger = logging.getLogger(__name__)

ImageInput = Union["PIL.Image.Image", np.ndarray, "torch.Tensor"]


class GetImagePatches(nn.Module):
    def __init__(
        self,
        possible_resolutions: List[Tuple[int, int]],
        patch_size: int,
        resample: Union[str, torchvision.transforms.InterpolationMode] = "bilinear",
    ) -> None:
        super().__init__()
        """
        Class used to divide and image with variable resolutions into patches of equal sizes,
        including the original image resized to [patch_size, patch_size].

        Forward returns a tensor of shape [num_patches + 1, channels, patch_size, patch_size].

        Args:
            possible_resolutions (List[Tuple[int, int]]): List of possible resolutions as [height, width].
            patch_size (int): Size of the patches to divide the image into.
            resample (Union[str, torchvision.transforms.InterpolationMode]):
            Resampling method, either 'bilinear', 'bicubic' or torchvision.transforms.InterpolationMode.
        """

        self.possible_resolutions = torch.tensor(possible_resolutions)

        if isinstance(resample, torchvision.transforms.InterpolationMode):
            self.resample = resample
        if resample == "bilinear":
            self.resample = torchvision.transforms.InterpolationMode.BILINEAR
        elif resample == "bicubic":
            self.resample = torchvision.transforms.InterpolationMode.BICUBIC
        else:
            raise ValueError(
                "resample must be of type torchvision.transforms.InterpolationMode or ['bilinear', 'bicubic]."
            )

        self.center_crop = v2.CenterCrop(size=patch_size)

        self.patch_size = patch_size

    @staticmethod
    def _get_max_res_without_distortion(
        image_size: Tuple[int, int],
        target_resolution: List[int],
    ) -> Tuple[int, int]:

        """
        Determines the maximum dimensions to which an image can be resized without distorting its
        aspect ratio, based on the target resolution.

        Args:
            image_size (Tuple[int, int]): The original dimensions of the image (height, width).
            target_resolution (List[int]): The desired resolution to fit the image into (height, width).
        Returns:
            Tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.
        Example:
            >>> _get_max_res_without_distortion([200, 300], target_resolution = [450, 200])
            (134, 200)
            >>> _get_max_res_without_distortion([800, 600], target_resolution = [450, 1300])
            (450, 338)
        """

        original_height, original_width = image_size
        target_height, target_width = target_resolution

        scale_w = target_width / original_width
        scale_h = target_height / original_height

        if scale_w < scale_h:
            new_width = target_width
            new_height = min(math.ceil(original_height * scale_w), target_height)
        else:
            new_height = target_height
            new_width = min(math.ceil(original_width * scale_h), target_width)

        return new_height, new_width

    @staticmethod
    def _divide_to_patches(image: torch.Tensor, patch_size: int) -> List[torch.Tensor]:
        """
        Divides an image into patches. The patches will not have equal size if the
        aspect ratio of the image is not divisible by the patch size.

        Args:
            image (torch.Tensor): Image tensor.
            patch_size (int): Size of each patch.
        Returns:
            List[torch.Tensor]: List of image patches.
        Example:
            >>> image = torch.rand(3, 200, 300)
            >>> patches = _divide_to_patches(image, patch_size=50)
            >>> len(patches)
            24
            >>> image = torch.rand(3, 400, 500)
            >>> patches = _divide_to_patches(image, patch_size=450)
            >>> len(patches)
            2
        """

        channel_size, height, width = image.shape

        patches = []
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch = image[:, i : i + patch_size, j : j + patch_size]
                patches.append(patch)

        return patches

    @staticmethod
    def _select_best_resolution(
        original_size: Tuple[int, int], possible_resolutions: torch.Tensor
    ) -> List[int]:
        """
        Selects the best resolution from a list of possible resolutions based on the original size.

        This is done by calculating the effective and wasted resolution for each possible resolution.

        The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

        Args:
            original_size (Tuple[int, int]):
                The original size of the image in the format [height, width].
            possible_resolutions (torch.Tensor):
                A list of possible resolutions in the format [[height1, width1], [height2, width2], ...].

        Returns:
            list: The best fit resolution in the format [height, width].

        Example:
            >>> _select_best_resolution((200, 300), torch.tensor[[100, 100], [200, 200], [300, 300], [400, 400]])
            [300, 300]
            >>> _select_best_resolution((800, 600), torch.tensor[[600, 800], [1600, 1200], [80, 60]])
            [1600, 1200]
        """

        original_height, original_width = original_size
        heights, widths = possible_resolutions[:, 0], possible_resolutions[:, 1]

        # Calculate the effective resolution and wasted resolution for each possible resolution
        # rescaling_factor is the minimum of the two scaling factors. Else one side would be outside of the canvas.
        rescaling_factor = torch.min(widths / original_width, heights / original_height)
        downscaled_widths = (original_width * rescaling_factor)
        downscaled_heights = (original_height * rescaling_factor)

        effective_resolutions = torch.min(
            downscaled_widths * downscaled_heights,
            torch.tensor(original_width * original_height),
        ).int()
        wasted_resolutions = (widths * heights) - effective_resolutions

        # Find the index of the best resolution
        max_effective_resolution = torch.max(effective_resolutions)
        indices_of_max = effective_resolutions == max_effective_resolution
        min_wasted_resolution = torch.min(wasted_resolutions[indices_of_max])

        best_index = torch.where(
            indices_of_max & (wasted_resolutions == min_wasted_resolution)
        )[0][0]
        best_fit: Tuple[int, int] = possible_resolutions[best_index].tolist()

        return best_fit

    @staticmethod
    def _get_padding_image_center(
        image_size: Tuple[int, int], target_resolution: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Determines the padding to be applied around the image to fit the target resolution.

        Args:
            image_size (Tuple[int, int]): The original size of the image in the format [height, width].
            target_resolution (Tuple[int, int]): The desired resolution to fit the image into in the format [height, width].

        Returns:
            Tuple[int, int]: The padding to be applied to the image in the format [top, bottom, left, right].
        """

        height, width = image_size
        target_height, target_width = target_resolution

        pad_x = (target_width - height) / 2
        pad_y = (target_height - width) / 2

        return (
            math.ceil(pad_y),
            math.ceil(pad_x),
            math.floor(pad_y),
            math.floor(pad_x),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:

        _, height, width = image.shape
        image_size = (height, width)

        best_resolution = self._select_best_resolution(
            image_size, self.possible_resolutions
        )

        # resize to closest best_resolution while preserving aspect ratio
        size_prepadding = self._get_max_res_without_distortion(
            image_size, best_resolution
        )

        resized_image = F.resize(image, size_prepadding, interpolation=self.resample)

        # pad to fit the best resolution
        padding = self._get_padding_image_center(
            image_size=size_prepadding, target_resolution=best_resolution
        )
        padded_image = F.pad(inpt=resized_image, padding=padding)

        # divide into patches, which may be of variable sizes
        image_patches = self._divide_to_patches(
            padded_image, patch_size=self.patch_size
        )

        # all patches are resized so that smallest side is self.patch_size.
        # this allows us to crop the images to [self.patch_size, self.patch_size]
        image_patches = [
            F.resize(patch, [self.patch_size], interpolation=self.resample).float()
            for patch in image_patches
        ]

        # Cropping only happens if the patch is not already a square. For example:
        # the image is resized and padded to best_resolution 400x300.
        # Assuming patch size is 200, the generated patches would be of size [200x200, 200x100, 200x200, 200x100]
        # The patches are then resized to have smallest_side = patch_size.
        # So the second and last patches would be resized to 400x200, and finally
        # center_cropped to 200x200. First and third patches are not cropped.
        image_patches = [self.center_crop(patch) for patch in image_patches]

        # original image is resized, with distortion, to a square of side self.patch_size
        # and concatenated to patches
        patch_sized_image = F.resize(
            image,
            [self.patch_size, self.patch_size],
            interpolation=self.resample,
        ).float()

        image_patches = torch.stack([patch_sized_image] + image_patches)

        return image_patches


class ImageProcessor(nn.Module):
    def __init__(
        self,
        patch_size: int = 336,
        possible_resolutions: Optional[Tuple[Tuple[int, int]]] = None,
        max_num_chunks: int = 4,
        resample: str = "bicubic",
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, Tuple[float]]] = None,
        image_std: Optional[Union[float, Tuple[float]]] = None,
    ) -> None:
        """
        Args:
            patch_size (int): Size of the patches to divide the image into.
            possible_resolutions (Optional[List[Tuple[int, int]]]): List of possible resolutions as tuples (height, width).
            max_num_chunks (int): Only used possible_resolutions is NOT given. Maximum number of chunks for processing
            high-resolution images.
            resample (str): Resampling method used when resizing images.
            do_rescale (bool): Flag to determine whether to rescale the image by `rescale_factor`.
            rescale_factor (Union[int, float]): Scale factor used if rescaling the image.
            do_normalize (bool): Flag to determine whether to normalize the image.
            image_mean (Optional[Union[float, List[float]]]): Mean values for normalization.
            image_std (Optional[Union[float, List[float]]]): Standard deviation values for normalization.
        """
        super().__init__()

        logger.info("Initializating ImageProcessor...")

        if image_mean is None:
            image_mean = [0.48145466, 0.4578275, 0.40821073]
        if image_std is None:
            image_std = [0.26862954, 0.26130258, 0.27577711]

        # If possible_resolutions are not given, then calculate possible ones based on max_num_chunks
        if not possible_resolutions:
            possible_resolutions = find_supported_resolutions(
                max_num_chunks=max_num_chunks, patch_size=patch_size
            )
        else:
            possible_resolutions = possible_resolutions

        logger.info(f"possible_resolutions: {possible_resolutions}")

        self.patchfy = GetImagePatches(
            possible_resolutions=possible_resolutions,
            patch_size=patch_size,
            resample=resample,
        )

        _preprocess = []
        if do_rescale:
            _preprocess.append(v2.Normalize(mean=[0] * 3, std=[1 / rescale_factor] * 3))

        if do_normalize:
            _preprocess.append(v2.Normalize(mean=image_mean, std=image_std))

        self._preprocess = nn.Sequential(*_preprocess)

    def preprocess(
        self, image: ImageInput
    ) -> Dict[str, torch.Tensor | Tuple[int, int]]:

        # Make image have dimension [3, H, W]. Input can be grayscale, RGB, channels-first or last.
        image = F.grayscale_to_rgb_image(F.to_image(image))
        _, height, width = image.shape

        patches = self.patchfy(image)

        patches = self._preprocess(patches)

        return {"pixel_values": patches, "image_size": (height, width)}
