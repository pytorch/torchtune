from typing import Dict, List, Optional, Union, Tuple

import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from functools import reduce

import math


def select_best_resolution(
        original_size: tuple, possible_resolutions: list
    ) -> tuple:
        """
        Selects the best resolution from a list of possible resolutions based on the original size.

        This is done by calculating the effective and wasted resolution for each possible resolution.

        The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

        Args:
            original_size (tuple):
                The original size of the image in the format (height, width).
            possible_resolutions (list):
                A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

        Returns:
            tuple: The best fit resolution in the format (height, width).
        """
        original_height, original_width = original_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float("inf")

        for height, width in possible_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = int(original_width * scale), int(
                original_height * scale
            )
            effective_resolution = min(
                downscaled_width * downscaled_height, original_width * original_height
            )
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution
            ):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (height, width)

        return best_fit

def find_closest_aspect_ratio(image_size, patch_size, aspect_ratios):
    image_width, image_height = image_size
    target_aspect_ratio = image_width / image_height
    closest_pair = None

    if target_aspect_ratio >= 1:
        # Handling landscape or square orientations
        closest_pair = min(
            [ratio for ratio in aspect_ratios.keys() if ratio <= target_aspect_ratio],
            key=lambda ratio: abs(ratio - target_aspect_ratio),
        )
        aspect_pairs = aspect_ratios[closest_pair]
        # Select the pair with the maximum width
        width_based_pairs = [(index, patch_size * width) for index, (width, _) in enumerate(aspect_pairs)]
        target_index = max(width_based_pairs, key=lambda x: x[1])[0]
    else:
        # Handling portrait orientations
        closest_pair = min(
            [ratio for ratio in aspect_ratios.keys() if ratio > target_aspect_ratio],
            key=lambda ratio: abs(1 / ratio - 1 / target_aspect_ratio),
        )
        aspect_pairs = aspect_ratios[closest_pair]
        # Select the pair with the maximum height
        height_based_pairs = [(index, patch_size * height) for index, (_, height) in enumerate(aspect_pairs)]
        target_index = max(height_based_pairs, key=lambda x: x[1])[0]
    selected_pair = aspect_pairs[target_index]
    return selected_pair

def find_supported_resolutions(max_num_chunks: int, patch_size: int):
        """
        This function computes all the allowed aspect ratios for a fixed
        number of input chunks.

        For example, with `num_chunks=5`, it will return:
        {
            0.2: [(1, 5)],
            5.0: [(5, 1)],
            0.25: [(1, 4)],
            1.0: [(2, 2), (1, 1)],
            4.0: [(4, 1)],
            0.3333333333333333: [(1, 3)],
            3.0: [(3, 1)],
            0.5: [(1, 2)],
            2.0: [(2, 1)]
        }
        """
        asp_dict = {}
        for chunk_size in range(max_num_chunks, 0, -1):
            _factors = sorted(factors(chunk_size))
            _asp_ratios = [(x, chunk_size // x) for x in _factors]
            for ratio in _asp_ratios:
                k = ratio[0] / ratio[1]
                if k not in asp_dict:
                    asp_dict[k] = [ratio]
                else:
                    asp_dict[k].append(ratio)

        possible_resolutions = []
        for key, value in asp_dict.items():
            for height, depth in value:
                possible_resolutions.append((height*patch_size, depth*patch_size))
        return possible_resolutions

def factors(n: int):
        """Return all factors of a number."""
        return set(
            reduce(
                list.__add__,
                ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
            )
        )



def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    From https://github.com/huggingface/transformers/blob/75f15f39a0434fe7a61385c4677f2700542a7ba6/src/transformers/image_processing_utils.py#L751C1-L786C20

    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """

    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit

def get_new_size_without_distortion(
        image_size: Tuple[int, int],
        target_resolution: Tuple[int, int],
        resize_strategy = "min_difference",
    ) -> Tuple[int, int]:

    """
    This function determines the maximum dimensions to which an image can be resized without distorting its
    aspect ratio, based on the target resolution.

    Args:
        image_size (Tuple[int, int]): The original dimensions of the image (height, width).
        target_resolution (Tuple[int, int]): The desired resolution to fit the image into (height, width).
        strategy (str): The strategy to use when determining the maximum dimensions. Options are "max_size" or "min_difference".
    Returns:
        Tuple[int, int]: The optimal dimensions (height, width) to which the image should be resized.

    Example 1:
    --------
     >>> original_size = (800, 600)
    >>> target_size = (1600, 800)
    >>> get_resize_without_distortion(original_size, target_size, resize_strategy = "min_difference")
    (1067, 800)

    Example 2:
    >>> original_size = (800, 600)
    >>> target_size = (1600, 800)
    >>> get_resize_without_distortion(original_size, target_size, resize_strategy = "max_size")
    (1600, 1200)
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

def divide_to_patches(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    channel_size, height, width = image.shape

    # Reshape to split height and width into patch_size blocks
    patches_height = height // patch_size
    patches_width = width // patch_size

    print(channel_size, height, width)
    print(patches_height, patches_width)
    reshaped = image.view(channel_size, patches_height, patch_size, patches_width, patch_size)

    # Transpose to bring patches together
    # We want [patches_height, patches_width, channel_size, patch_size, patch_size]
    transposed = reshaped.permute(1, 3, 0, 2, 4)

    # Flatten the patches
    patches = transposed.contiguous().view(patches_height * patches_width, channel_size, patch_size, patch_size)
    return patches

class GetImagePatches(nn.Module):
    def __init__(self,
        possible_resolutions,
        patch_size: int = 224,
        resample: str = 'bicubic',
        keep_original_and_resize: bool = False,
    ) -> torch.Tensor:
        super().__init__()
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            possible_resolutions (List):
                A string representation of a list of possible resolutions.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`str`):
                Either 'bilinear' or 'bicubic'
            keep_original_and_resize (`bool`):
                Whether to keep the original image in the output.

        Returns:
            torch.Tensor:
                A tensor of shape [num_patches, channels, patch_size, patch_size] if keep_original_and_resize is False,
                otherwise [num_patches + 1, channels, patch_size, patch_size].
        """

        if not isinstance(possible_resolutions, Union[List, Tuple]):
            raise ValueError(f"Possible_resolutions must be a List[List[int, int]]. Got {possible_resolutions}")

        self.possible_resolutions = possible_resolutions

        if resample == "bilinear":
            self.resample = torchvision.transforms.InterpolationMode.BILINEAR
        elif resample == "bicubic":
            self.resample = torchvision.transforms.InterpolationMode.BICUBIC
        else:
            raise ValueError(f"resample must be either 'bilinear' or 'bicubic'. Got {resample}.")

        self.keep_original_and_resize = keep_original_and_resize
        self.patch_size = patch_size

    def forward(self, image: torch.Tensor):

        image_size = F.get_image_size(image)
        best_resolution = select_best_resolution(image_size, self.possible_resolutions)

        # resize while preserving aspect ratio
        size_prepadding = get_resize_without_distortion(image_size, best_resolution)

        resized_image = F.resize(
            image, size_prepadding, interpolation=self.resample
        )

        # pad to fit the best resolution
        pad_x, pad_y = (best_resolution[0] - size_prepadding[0])//2, (best_resolution[1] - size_prepadding[1])//2
        padded_image = F.pad(resized_image, [pad_x, pad_y])

        # divide into patches
        image_patches = divide_to_patches(padded_image, patch_size=self.patch_size)

        if self.keep_original_and_resize:
            patch_sized_image = F.resize(image, [self.patch_size, self.patch_size], interpolation=self.resample)
            image_patches = torch.stack([image_patches + patch_sized_image])

        return image_patches
