# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtune.utils.image_transforms_utils import find_supported_resolutions


class TestFindSupportedResolutions:
    @pytest.mark.parametrize(
        "max_num_chunks, patch_size, expected",
        [
            (1, 10, [(10, 10)]),
            (2, 7, [(7, 14), (14, 7), (7, 7)]),
            (3, 8, [(8, 24), (24, 8), (8, 16), (16, 8), (8, 8)]),
        ],
    )
    def test_find_supported_resolutions(self, max_num_chunks, patch_size, expected):
        output_resolutions = find_supported_resolutions(max_num_chunks, patch_size)

        for output in output_resolutions:
            # assert max num chunks per resoltuion
            num_chunks_h = output[0] / patch_size
            num_chunks_w = output[1] / patch_size
            assert (
                num_chunks_h * num_chunks_w <= max_num_chunks
            ), f"Expected {max_num_chunks} but got {num_chunks_h * num_chunks_w}"

            # asserts resolution is a multiple of patch size
            assert (
                output[0] % patch_size == 0
            ), f"Expected height {output[0]} to be a multiple of {patch_size} but got {output[0] % patch_size}"
            assert (
                output[1] % patch_size == 0
            ), f"Expected width {output[1]} to be a multiple of {patch_size} but got {output[1] % patch_size}"

        # matches expected
        assert sorted(output_resolutions) == sorted(
            expected
        ), f"Expected {sorted(expected)} but got {sorted(output_resolutions)}"

        # all unique
        assert len(output_resolutions) == len(
            set(output_resolutions)
        ), f"Expected {len(set(output_resolutions))} but got {len(output_resolutions)}"
