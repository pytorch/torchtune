# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import PIL
import pytest

import torch
from tests.test_utils import assert_expected

from torchtune.models.clip._transforms import CLIPImageTransform


class TestCLIPImageTransform:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (100, 400, 3),
                "expected_shape": torch.Size([2, 3, 224, 224]),
                "resize_to_max_canvas": False,
                "expected_tile_means": [0.2231, 0.1754],
                "expected_aspect_ratio": [1, 2],
            },
            {
                "image_size": (1000, 300, 3),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": True,
                "expected_tile_means": [0.4999, 0.5000, 0.5000, 0.1652],
                "expected_aspect_ratio": [4, 1],
            },
            {
                "image_size": (200, 200, 3),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": True,
                "expected_tile_means": [0.4991, 0.5000, 0.4997, 0.5004],
                "expected_aspect_ratio": [2, 2],
            },
            {
                "image_size": (600, 200, 3),
                "expected_shape": torch.Size([3, 3, 224, 224]),
                "resize_to_max_canvas": False,
                "expected_tile_means": [0.4464, 0.4464, 0.3028],
                "expected_aspect_ratio": [3, 1],
            },
        ],
    )
    def test_clip_image_transform(self, params):
        # Initialize the image transformation with specified parameters
        image_transform = CLIPImageTransform(
            image_mean=None,
            image_std=None,
            tile_size=224,
            possible_resolutions=None,
            max_num_tiles=4,
            resample="bilinear",
            resize_to_max_canvas=params["resize_to_max_canvas"],
        )

        # Generate a deterministic image using np.arange for reproducibility
        image_size = params["image_size"]
        image = (np.arange(np.prod(image_size)).reshape(image_size)).astype(np.uint8)
        image = PIL.Image.fromarray(image)

        # Apply the transformation
        output = image_transform(image=image)
        output_image = output["image"]
        output_ar = output["aspect_ratio"]

        # output shape matches the expected shape
        assert (
            output_image.shape == params["expected_shape"]
        ), f"Expected shape {params['expected_shape']} but got {output_image.shape}"

        # pixel values are within the expected range [0, 1]
        assert (
            0 <= output_image.min() <= output_image.max() <= 1
        ), f"Expected pixel values to be in range [0, 1] but got {output_image.min()} and {output_image.max()}"

        #  mean values of the tiles match the expected means
        for i, tile in enumerate(output_image):
            assert_expected(
                tile.mean().item(), params["expected_tile_means"][i], rtol=0, atol=1e-4
            )

        #  aspect ratio matches the expected aspect ratio
        assert tuple(output_ar.numpy()) == tuple(
            params["expected_aspect_ratio"]
        ), f"Expected aspect ratio {params['expected_aspect_ratio']} but got {tuple(output_ar.numpy())}"

        # number of tiles matches the product of the aspect ratio
        expected_num_tiles = output_ar[0] * output_ar[1]
        assert (
            expected_num_tiles == output_image.shape[0]
        ), f"Expected {expected_num_tiles} tiles but got {output_image.shape[0]}"
