from typing import Dict, List, Optional, Union, Tuple

import torch
from torch import nn
import torchvision
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from functools import reduce

import math

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
