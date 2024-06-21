# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchtune.modules.transforms.vision import tile_crop


class TestTransforms:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (3, 200, 300),
                "tile_size": 50,
                "num_tiles": 24,
                "tile_shape": (50, 50),
                "status": "Passed",
            },
            {
                "image_size": (3, 400, 600),
                "tile_size": 200,
                "num_tiles": 6,
                "tile_shape": (200, 200),
                "status": "Passed",
            },
            {
                "image_size": (3, 250, 250),
                "tile_size": 250,
                "num_tiles": 1,
                "tile_shape": (250, 250),
                "status": "Passed",
            },
            {
                "image_size": (3, 250, 250),
                "tile_size": 500,
                "status": "Failed",
                "error": "shape '[3, 0, 500, 0, 500]' is invalid for input of size 187500",
            },
            {
                "image_size": (3, 250, 250),
                "tile_size": 80,
                "status": "Failed",
                "error": "shape '[3, 3, 80, 3, 80]' is invalid for input of size 187500",
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
            assert (
                tiles.shape[0] == params["num_tiles"]
            ), f"Expected number of tiles {params['num_tiles']} but got {tiles.shape[0]}"
            assert (
                tiles.shape[-2:] == params["tile_shape"]
            ), f"Expected tile shape {params['tile_shape']} but got {tiles.shape[-2:]}"

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
            assert (
                str(exc_info.value) == params["error"]
            ), f"Expected error message '{params['error']}' but got '{str(exc_info.value)}'"
