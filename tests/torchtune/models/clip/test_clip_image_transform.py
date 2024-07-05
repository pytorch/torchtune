# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import PIL
import pytest

import torch

from torchtune.models.clip._transforms import CLIPImageTransform


class TestPipelines:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (100, 100, 3),
                "expected_shape": torch.Size([1, 3, 224, 224]),
                "resize_to_max_canvas": False,
            },
            {
                "image_size": (200, 300),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": True,
            },
            {
                "image_size": (100, 200, 3),
                "expected_shape": torch.Size([2, 3, 224, 224]),
                "resize_to_max_canvas": True,
            },
            {
                "image_size": (100, 200),
                "expected_shape": torch.Size([1, 3, 224, 224]),
                "resize_to_max_canvas": False,
            },
        ],
    )
    def test_clip_image_transform(self, params):

        image_transform = CLIPImageTransform(
            image_mean=None,
            image_std=None,
            tile_size=224,
            possible_resolutions=None,
            max_num_tiles=4,
            resample="bilinear",
            resize_to_max_canvas=params["resize_to_max_canvas"],
        )

        image_size = params["image_size"]

        # Create a random image
        image = (np.random.rand(*image_size) * 255).astype(np.uint8)  # type: ignore
        image = PIL.Image.fromarray(image)  # type: ignore

        output = image_transform(image=image)
        output_image = output["image"]
        output_ar = output["aspect_ratio"]

        assert (
            output_image.shape == params["expected_shape"]
        ), f"Expected shape {params['expected_shape']} but got {output_image.shape}"

        assert (
            0 <= output_image.min() <= output_image.max() <= 1
        ), f"Expected pixel values to be in range [0, 1] but got {output_image.min()} and {output_image.max()}"

        expected_num_tiles = output_ar[0] * output_ar[1]
        assert (
            expected_num_tiles == output_image.shape[0]
        ), f"Expected {expected_num_tiles} tiles but got {output_image.shape[0]}"
