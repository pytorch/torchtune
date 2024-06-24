# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import PIL
import pytest

import torch

from torchtune.models.clip._component_builders import CLIPImageTransform


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
                "expected_shape": torch.Size([2, 3, 224, 224]),
                "resize_to_max_canvas": True,
            },
            {
                "image_size": (400, 400, 3),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": True,
            },
            {
                "image_size": (800, 600),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": False,
            },
        ],
    )
    def test_shapes_variable_image_size_transforms(self, params, image_transform):

        image_transform = CLIPImageTransform(
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
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

        output = image_transform(image)
        pixel_values = output["pixel_values"]

        assert (
            pixel_values.shape == params["expected_shape"]
        ), f"Expected shape {params['expected_shape']} but got {pixel_values.shape}"

        assert (
            0 <= pixel_values.min() <= pixel_values.max() <= 1
        ), f"Expected pixel values to be in range [0, 1] but got {pixel_values.min()} and {pixel_values.max()}"
