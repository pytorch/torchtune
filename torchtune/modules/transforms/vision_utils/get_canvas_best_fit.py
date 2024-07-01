# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from collections import defaultdict
from typing import List, Set, Tuple

import torch

logger = logging.getLogger(__name__)


def get_canvas_best_fit(
    image: torch.Tensor, possible_resolutions: torch.Tensor, resize_to_max_canvas: bool
) -> Tuple[int, int]:
    """
    Determines the best canvas possible from a list of possible resolutions to
    resize an image to, without distortion.

    For each possible resolution, calculates the scaling factors for
    width and height, and selects the smallest one, which is the limiting side.
    E.g. if to match a canvas shape you have to upscale an image's height by 2x, and width by 1.5x,
    then the maximum upscaling without distortion is min(2, 1.5) = 1.5.

    If there are multiple canvases that satisfy the conditions,
    we pick the one with the lowest area to minimize padding.

    Args:
        image (torch.Tensor): The image we want to fit into a canvas.
        possible_resolutions (torch.Tensor): A tensor of shape (N, 2) where each
            row represents a possible canvas.
        resize_to_max_canvas (bool): If True, pick the canvas that allows maximum scaling.
            If False, pick the canvas that minimizes downscaling, including no downscaling at all.

    Returns:
        Tuple[int, int]: The best resolution to fit the image into.

    Examples:
        >>> image = torch.rand(3, 200, 300)
        >>> possible_resolutions = torch.tensor([
        ...     [224, 672],
        ...     [672, 224],
        ...     [224, 448],
        ...     [448, 224],
        ...     [224, 224]
        ... ])
        >>> get_canvas_best_fit(image, possible_resolutions, resize_to_max_canvas=False)
        (224, 448)

        In the example above, we calculate the scaling factors for each possible resolution

        >>> scale_height = torch.tensor([1.1200, 3.3600, 1.1200, 2.2400, 1.1200])
        >>> scale_width = torch.tensor([2.2400, 0.7467, 1.4933, 0.7467, 0.7467])
        >>> scales = torch.tensor([1.1200, 0.7467, 1.1200, 0.7467, 0.7467])

        Two options have scaling_factor > 1, since resize_to_max_canvas is False, we pick the smallest

        >>> upscaling_options = torch.tensor([1.1200, 1.1200])
        >>> selected_scale = torch.tensor(1.1200)

        There are two possible options, so we pick the one with the smallest area

        >>> areas = torch.tensor([150528, 100352])  # for resolutions [672, 224] and [224, 448], respectively
        >>> optimal_canvas = torch.tensor([224, 448])  # resolution with the smallest area
    """

    original_height, original_width = image.shape[-2:]

    # possible resolutions heights/widths
    target_heights, target_widths = (
        possible_resolutions[:, 0],
        possible_resolutions[:, 1],
    )

    # scaling factors to resize the image without distortion
    scale_w = target_widths / original_width
    scale_h = target_heights / original_height

    # get limiting side scaling -> no distortion
    scales = torch.where(scale_w > scale_h, scale_h, scale_w)

    # filter only scales that allow upscaling
    upscaling_options = scales[scales >= 1]
    if len(upscaling_options) > 0:
        if resize_to_max_canvas:
            selected_scale = torch.max(upscaling_options)
        else:
            selected_scale = torch.min(upscaling_options)
    else:
        # no upscaling possible,
        # get the minimum downscaling (max scale for scales<1)
        downscaling_options = scales[scales < 1]
        selected_scale = torch.max(downscaling_options)

    # get all resolutions that support this scaling factor,
    # e.g. you can upscale to 224x224, 224x448, 224x672 without distortion
    chosen_canvas = possible_resolutions[scales == selected_scale]

    # if there are multiple resolutions,
    # get the one with minimum area to reduce padding
    if len(chosen_canvas) > 1:
        areas = chosen_canvas[:, 0] * chosen_canvas[:, 1]
        optimal_idx = torch.argmin(areas)
        optimal_canvas = chosen_canvas[optimal_idx]
    else:
        optimal_canvas = chosen_canvas[0]

    return tuple(optimal_canvas.tolist())


def find_supported_resolutions(
    max_num_tiles: int, tile_size: int
) -> List[Tuple[int, int]]:
    """
    Computes all combinations of resolutions, multiple of tile_size,
    that contain up to max_num_tiles. Useful for when dividing an image into tiles.

    For example, if we want at most 2 tiles per image, then we can support the
    following resolutions: (1x1, 1x2, 2x1) * tile_size

    Args:
        max_num_tiles (int): Maximum number of tiles.
        tile_size (int): Size of the side of the tile.

    Returns:
        List[Tuple[int, int]]: List of possible resolutions as tuples (height, width).

    Examples:

        >>> max_num_tiles = 4
        >>> tile_size = 224
        >>> find_supported_resolutions(max_num_tiles, tile_size)
        [(224, 896), (448, 448), (224, 224), (896, 224), (224, 672), (672, 224), (224, 448), (448, 224)]
    """

    # create dictionary {aspect_ratio: [resolution1, ..., resolution n]}
    # example {0.25: [(1,4)], 1.0: [(2,2), (1,1)], 4.0: [(4,1)]}
    asp_dict = defaultdict(list)
    for _tile_size in range(max_num_tiles, 0, -1):
        factors = sorted(_get_factors(_tile_size))
        asp_ratios = [(factor, _tile_size // factor) for factor in factors]
        for height, width in asp_ratios:
            ratio_float = height / width
            asp_dict[ratio_float].append((height, width))

    # get the resolutions multiplied by the tile_size
    possible_resolutions = []
    for ar, resolution in asp_dict.items():
        for height, width in resolution:
            possible_resolutions.append((height * tile_size, width * tile_size))

    return possible_resolutions


def _get_factors(n: int) -> Set[int]:
    """
    Calculate all factors of a given number, i.e. a divisor that leaves no remainder.

    Args:
        n (int): The number to find factors for.

    Returns:
        set: A set containing all factors of the number.

    Examples:
        >>> _get_factors(n=12)
        {1, 2, 3, 4, 6, 12}
    """
    factors_set = set()

    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors_set.add(i)
            factors_set.add(n // i)
    return factors_set
