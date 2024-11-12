# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch
import torchvision

from torchtune.modules.transforms.vision_utils.resize_with_pad import resize_with_pad


class TestTransforms:
    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (200, 100),
                "target_size": (1000, 1200),
                "max_size": 600,
                "expected_resized_size": (600, 300),
            },
            {
                "image_size": (2000, 200),
                "target_size": (1000, 1200),
                "max_size": 600,
                "expected_resized_size": (1000, 100),
            },
            {
                "image_size": (400, 200),
                "target_size": (1000, 1200),
                "max_size": 2000,
                "expected_resized_size": (1000, 500),
            },
            {
                "image_size": (400, 200),
                "target_size": (1000, 1200),
                "max_size": None,
                "expected_resized_size": (1000, 500),
            },
            {
                "image_size": (1000, 500),
                "target_size": (400, 300),
                "max_size": None,
                "expected_resized_size": [400, 200],
            },
        ],
    )
    def test_resize_with_pad(self, params):

        image_size = params["image_size"]
        target_size = params["target_size"]
        max_size = params["max_size"]
        expected_resized_size = params["expected_resized_size"]

        image = torch.rand(3, *image_size)  # Create a random image tensor

        resized_image = resize_with_pad(
            image=image,
            target_size=target_size,
            resample=torchvision.transforms.InterpolationMode["BILINEAR"],
            max_size=max_size,
        )

        # assert everything beyond resize has value == 0
        assert torch.all(
            resized_image[:, (expected_resized_size[0] + 1) :, :] == 0
        ), "Expected everything beyond resize to be pad with fill=0"

        assert torch.all(
            resized_image[:, :, (expected_resized_size[1] + 1) :] == 0
        ), "Expected everything beyond resize to be pad with fill=0"

        assert torch.all(
            resized_image[:, : expected_resized_size[0], : expected_resized_size[1]]
            != 0
        ), "Expected no padding where the image is supposed to be"

        # output should have shape target_size
        assert (
            resized_image.shape[-2:] == target_size
        ), f"Expected output with shape {target_size} but got {resized_image.shape[-2:]}"
