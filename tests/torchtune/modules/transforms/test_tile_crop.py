# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop


class TestTransforms:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "expected_output_shape": torch.Size([24, 3, 50, 50]),
                "image_size": (3, 200, 300),
                "status": "Passed",
                "tile_size": 50,
            },
            {
                "expected_output_shape": torch.Size([6, 3, 200, 200]),
                "image_size": (3, 400, 600),
                "status": "Passed",
                "tile_size": 200,
            },
            {
                "expected_output_shape": torch.Size([1, 3, 250, 250]),
                "image_size": (3, 250, 250),
                "status": "Passed",
                "tile_size": 250,
            },
            {
                "error": "Image size 250x250 is not divisible by tile size 500",
                "image_size": (3, 250, 250),
                "status": "Failed",
                "tile_size": 500,
            },
            {
                "error": "Image size 250x250 is not divisible by tile size 80",
                "image_size": (3, 250, 250),
                "status": "Failed",
                "tile_size": 80,
            },
        ],
    )
    def test_tile_crop(self, params):
        image_size = params["image_size"]
        tile_size = params["tile_size"]
        status = params["status"]

        image = torch.rand(*image_size)  # Create a random image tensor

        if status == "Passed":
            tiles = tile_crop(image, tile_size)
            expected_output_shape = params["expected_output_shape"]
            assert (
                tiles.shape == expected_output_shape
            ), f"Expected shape {expected_output_shape} but got {tiles.shape}"

            # check if first and last tile matches the image
            first_tile = image[..., :tile_size, :tile_size]
            last_tile = image[..., -tile_size:, -tile_size:]
            assert torch.equal(
                tiles[0], first_tile
            ), "Expected first tile to match the image"
            assert torch.equal(
                tiles[-1], last_tile
            ), "Expected last tile to match the image"

        elif status == "Failed":
            with pytest.raises(Exception) as exc_info:
                tile_crop(image, tile_size)
            expected_error = params["error"]
            actual_error = str(exc_info.value)
            assert (
                str(exc_info.value) == params["error"]
            ), f"Expected error message '{expected_error}' but got '{actual_error}'"
