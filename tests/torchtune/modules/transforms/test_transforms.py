# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from torchtune.modules.transforms.transforms import (
    divide_to_equal_patches,
    pad_image_top_left,
    ResizeWithoutDistortion,
)


class TestTransforms:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (3, 10, 10),
                "target_size": (20, 20),
                "expected_padded_size": (20, 20),
            },
            {
                "image_size": (1, 30, 40),
                "target_size": (30, 40),
                "expected_padded_size": (30, 40),
            },
            {
                "image_size": (3, 50, 30),
                "target_size": (60, 30),
                "expected_padded_size": (60, 30),
            },
            {
                "image_size": (5, 2, 100, 150),
                "target_size": (100, 100),
                "expected_padded_size": (100, 100),
            },
            {
                "image_size": (3, 200, 100),
                "target_size": (200, 200),
                "expected_padded_size": (200, 200),
            },
        ],
    )
    def test_pad_image_top_left(self, params):
        image_size = params["image_size"]
        target_size = params["target_size"]
        expected_padded_size = params["expected_padded_size"]

        image = torch.rand(*image_size)  # Create a random image tensor
        padded_image = pad_image_top_left(image, target_size)

        # assert shapes
        assert (
            padded_image.shape[-2:] == expected_padded_size
        ), f"Expected padded size {expected_padded_size} but got {padded_image.shape[-2:]}"

        # assert the non-padded pixels are equal in both images
        height_size = min(image_size[-2], target_size[-2])
        width_size = min(image_size[-1], target_size[-1])
        assert torch.equal(
            padded_image[..., :height_size, :width_size],
            image[..., :height_size, :width_size],
        ), "Expected the non-padded pixels to be equal in both images"

    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (200, 100),
                "target_size": (1000, 1200),
                "max_upscaling_size": 600,
                "expected_resized_size": (600, 300),
            },
            {
                "image_size": (2000, 200),
                "target_size": (1000, 1200),
                "max_upscaling_size": 600,
                "expected_resized_size": (1000, 100),
            },
            {
                "image_size": (400, 200),
                "target_size": (1000, 1200),
                "max_upscaling_size": 2000,
                "expected_resized_size": (1000, 500),
            },
            {
                "image_size": (400, 200),
                "target_size": (1000, 1200),
                "max_upscaling_size": None,
                "expected_resized_size": (1000, 500),
            },
            {
                "image_size": (1000, 500),
                "target_size": (400, 300),
                "max_upscaling_size": None,
                "expected_resized_size": torch.Size([400, 200]),
            },
        ],
    )
    def test_resize_without_distortion(self, params):

        image_size = params["image_size"]
        target_size = params["target_size"]
        max_upscaling_size = params["max_upscaling_size"]
        expected_resized_size = params["expected_resized_size"]

        image = torch.rand(3, *image_size)  # Create a random image tensor
        resizer = ResizeWithoutDistortion(
            resample="bicubic", max_upscaling_size=max_upscaling_size
        )
        resized_image = resizer(image, target_size)
        assert (
            resized_image.shape[-2:] == expected_resized_size
        ), f"Expected resized size {expected_resized_size} but got {resized_image.shape[-2:]}"

    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (3, 200, 300),
                "patch_size": 50,
                "num_patches": 24,
                "patch_shape": (50, 50),
                "status": "Passed",
            },
            {
                "image_size": (3, 400, 600),
                "patch_size": 200,
                "num_patches": 6,
                "patch_shape": (200, 200),
                "status": "Passed",
            },
            {
                "image_size": (3, 250, 250),
                "patch_size": 250,
                "num_patches": 1,
                "patch_shape": (250, 250),
                "status": "Passed",
            },
            {
                "image_size": (3, 250, 250),
                "patch_size": 500,
                "status": "Failed",
                "error": "shape '[3, 0, 500, 0, 500]' is invalid for input of size 187500",
            },
            {
                "image_size": (3, 250, 250),
                "patch_size": 80,
                "status": "Failed",
                "error": "shape '[3, 3, 80, 3, 80]' is invalid for input of size 187500",
            },
        ],
    )
    def test_divide_to_equal_patches(self, params):
        image_size = params["image_size"]
        patch_size = params["patch_size"]
        status = params["status"]

        image = torch.rand(*image_size)  # Create a random image tensor

        if status == "Passed":
            patches = divide_to_equal_patches(image, patch_size)
            assert (
                patches.shape[0] == params["num_patches"]
            ), f"Expected number of patches {params['num_patches']} but got {patches.shape[0]}"
            assert (
                patches.shape[-2:] == params["patch_shape"]
            ), f"Expected patch shape {params['patch_shape']} but got {patches.shape[-2:]}"

            # check if first and last patch matches the image
            first_patch = image[..., :patch_size, :patch_size]
            last_patch = image[..., -patch_size:, -patch_size:]
            assert torch.equal(
                patches[0], first_patch
            ), "Expected first patch to match the image"
            assert torch.equal(
                patches[-1], last_patch
            ), "Expected last patch to match the image"

        elif status == "Failed":
            with pytest.raises(Exception) as exc_info:
                divide_to_equal_patches(image, patch_size)
            assert (
                str(exc_info.value) == params["error"]
            ), f"Expected error message '{params['error']}' but got '{str(exc_info.value)}'"
