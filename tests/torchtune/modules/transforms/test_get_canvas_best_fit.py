# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtune.modules.transforms.vision_utils.get_canvas_best_fit import (
    find_supported_resolutions,
    get_canvas_best_fit,
)


class TestUtils:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "max_num_tiles": 1,
                "tile_size": 224,
                "expected_resolutions": [(224, 224)],
            },
            {
                "max_num_tiles": 2,
                "tile_size": 100,
                "expected_resolutions": [(100, 200), (200, 100), (100, 100)],
            },
            {
                "max_num_tiles": 3,
                "tile_size": 50,
                "expected_resolutions": [
                    (50, 150),
                    (150, 50),
                    (50, 100),
                    (100, 50),
                    (50, 50),
                ],
            },
            {
                "max_num_tiles": 4,
                "tile_size": 300,
                "expected_resolutions": [
                    (300, 1200),
                    (600, 600),
                    (300, 300),
                    (1200, 300),
                    (300, 900),
                    (900, 300),
                    (300, 600),
                    (600, 300),
                ],
            },
        ],
    )
    def test_find_supported_resolutions(self, params):
        max_num_tiles = params["max_num_tiles"]
        tile_size = params["tile_size"]
        expected_resolutions = params["expected_resolutions"]
        resolutions = find_supported_resolutions(max_num_tiles, tile_size)

        assert len(set(resolutions)) == len(resolutions), "Resolutions should be unique"
        assert set(resolutions) == set(
            expected_resolutions
        ), f"Expected resolutions {expected_resolutions} but got {resolutions}"

    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (800, 600),
                "possible_resolutions": [
                    (224, 896),
                    (448, 448),
                    (224, 224),
                    (896, 224),
                    (224, 672),
                    (672, 224),
                    (224, 448),
                    (448, 224),
                ],
                "resize_to_max_canvax": False,
                "expected_best_resolution": (448, 448),
            },
            {
                "image_size": (200, 300),
                "possible_resolutions": [
                    (224, 896),
                    (448, 448),
                    (224, 224),
                    (896, 224),
                    (224, 672),
                    (672, 224),
                    (224, 448),
                    (448, 224),
                ],
                "resize_to_max_canvax": False,
                "expected_best_resolution": (224, 448),
            },
            {
                "image_size": (200, 500),
                "possible_resolutions": [
                    (224, 896),
                    (448, 448),
                    (224, 224),
                    (896, 224),
                    (224, 672),
                    (672, 224),
                    (224, 448),
                    (448, 224),
                ],
                "resize_to_max_canvax": True,
                "expected_best_resolution": (224, 672),
            },
            {
                "image_size": (200, 200),
                "possible_resolutions": [
                    (224, 896),
                    (448, 448),
                    (224, 224),
                    (896, 224),
                    (224, 672),
                    (672, 224),
                    (224, 448),
                    (448, 224),
                ],
                "resize_to_max_canvax": False,
                "expected_best_resolution": (224, 224),
            },
            {
                "image_size": (200, 100),
                "possible_resolutions": [
                    (224, 896),
                    (448, 448),
                    (224, 224),
                    (896, 224),
                    (224, 672),
                    (672, 224),
                    (224, 448),
                    (448, 224),
                ],
                "resize_to_max_canvax": True,
                "expected_best_resolution": (448, 224),
            },
        ],
    )
    def test_get_canvas_best_fit(self, params):
        image_size = params["image_size"]
        possible_resolutions = params["possible_resolutions"]
        expected_best_resolution = params["expected_best_resolution"]
        resize_to_max_canvax = params["resize_to_max_canvax"]

        possible_resolutions = torch.tensor(possible_resolutions)

        image = torch.rand(*image_size)
        best_resolution = get_canvas_best_fit(
            image, possible_resolutions, resize_to_max_canvax
        )

        assert (
            tuple(best_resolution) == expected_best_resolution
        ), f"Expected best resolution {expected_best_resolution} but got {best_resolution}"
