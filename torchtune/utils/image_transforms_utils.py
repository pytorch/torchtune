# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce
from typing import List, Tuple


def find_supported_resolutions(
    max_num_chunks: int, patch_size: int
) -> List[Tuple[int, int]]:
    """
    This function computes all the allowed aspect ratios for a fixed
    number of input chunks.

    For example, with `num_chunks=5`, it will create:
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

    and return the resolutions multiplied by the patch_size.
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
            possible_resolutions.append((height * patch_size, depth * patch_size))
    return possible_resolutions


def factors(n: int):
    """Return all factors of a number."""
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )
