# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import PIL
import pytest

import torch
from PIL import Image

from tests.common import ASSETS
from tests.test_utils import assert_expected

from torchtune.models.clip._transform import _load_image, CLIPImageTransform
from torchtune.models.clip.inference._transform import (
    CLIPImageTransform as CLIPImageTransformInference,
)
from torchtune.training.seed import set_seed


@pytest.fixture(autouse=True)
def random():
    set_seed(0)


class TestCLIPImageTransform:
    @pytest.fixture
    def tmp_image(self):
        return str(ASSETS / "dog_on_skateboard.jpg")

    def test_load_image_local_file(self, tmp_image):
        # Load the image
        image = _load_image(tmp_image)

        # Check that the image is loaded correctly
        assert isinstance(image, Image.Image)
        assert image.size == (580, 403)

    def test_load_image_remote_file(self, monkeypatch, tmp_image):
        # Mock the urlopen function to return a BytesIO object
        def mock_urlopen(url):
            return open(tmp_image, "rb")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

        # Load the image
        image = _load_image("http://example.com/test_image.jpg")

        # Check that the image is loaded correctly
        assert isinstance(image, Image.Image)
        assert image.size == (580, 403)

    def test_load_image_invalid_path(self):
        # Test that a ValueError is raised when the image path is invalid
        with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
            _load_image("invalid_path")

    def test_load_image_invalid_image_data(self, tmp_path):
        # Create a temporary file with invalid image data
        image_path = tmp_path / "test_image.jpg"
        with open(image_path, "w") as f:
            f.write("Invalid image data")

        # Test that a ValueError is raised when the image data is invalid
        with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
            _load_image(str(image_path))

    def test_load_image_http_error(self, monkeypatch):
        # Mock the urlopen function to raise an exception
        def mock_urlopen(url):
            raise Exception("Failed to load image")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

        # Test that a ValueError is raised when there is an HTTP error
        with pytest.raises(ValueError, match="Failed to load image"):
            _load_image("http://example.com/test_image.jpg")

    def test_load_image_io_error(self, tmp_path):
        # Create a temporary file that cannot be read
        image_path = tmp_path / "test_image.jpg"
        with open(image_path, "w") as f:
            f.write("Test data")
        os.chmod(image_path, 0o000)  # Remove read permissions

        # Test that a ValueError is raised when there is an IO error
        with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
            _load_image(str(image_path))
        os.chmod(image_path, 0o644)  # Restore read permissions

    def test_load_image_pil_error(self, tmp_path):
        # Create a temporary file with invalid image data
        image_path = tmp_path / "test_image.jpg"
        with open(image_path, "wb") as f:
            f.write(b"Invalid image data")

        # Test that a ValueError is raised when PIL cannot open the image
        with pytest.raises(ValueError, match="Failed to open image as PIL.Image"):
            _load_image(str(image_path))

    @pytest.mark.parametrize(
        "params",
        [
            {
                "image_size": (100, 400, 3),
                "expected_shape": torch.Size([2, 3, 224, 224]),
                "resize_to_max_canvas": False,
                "expected_tile_means": [0.2230, 0.1763],
                "expected_tile_max": [1.0, 1.0],
                "expected_tile_min": [0.0, 0.0],
                "expected_aspect_ratio": [1, 2],
            },
            {
                "image_size": (1000, 300, 3),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": True,
                "expected_tile_means": [0.5007, 0.4995, 0.5003, 0.1651],
                "expected_tile_max": [0.9705, 0.9694, 0.9521, 0.9314],
                "expected_tile_min": [0.0353, 0.0435, 0.0528, 0.0],
                "expected_aspect_ratio": [4, 1],
            },
            {
                "image_size": (200, 200, 3),
                "expected_shape": torch.Size([4, 3, 224, 224]),
                "resize_to_max_canvas": True,
                "expected_tile_means": [0.5012, 0.5020, 0.5011, 0.4991],
                "expected_tile_max": [0.9922, 0.9926, 0.9970, 0.9908],
                "expected_tile_min": [0.0056, 0.0069, 0.0059, 0.0033],
                "expected_aspect_ratio": [2, 2],
            },
            {
                "image_size": (600, 200, 3),
                "expected_shape": torch.Size([3, 3, 224, 224]),
                "resize_to_max_canvas": False,
                "expected_tile_means": [0.4473, 0.4469, 0.3032],
                "expected_tile_max": [1.0, 1.0, 1.0],
                "expected_tile_min": [0.0, 0.0, 0.0],
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
            dtype=torch.float32,
            resize_to_max_canvas=params["resize_to_max_canvas"],
        )

        image_transform_inference = CLIPImageTransformInference(
            image_mean=None,
            image_std=None,
            tile_size=224,
            possible_resolutions=None,
            max_num_tiles=4,
            resample="bilinear",
            resize_to_max_canvas=params["resize_to_max_canvas"],
            antialias=True,
        )

        # Generate a deterministic image using np.arange for reproducibility
        image_size = params["image_size"]
        image = (
            np.random.randint(0, 256, np.prod(image_size))
            .reshape(image_size)
            .astype(np.uint8)
        )
        image = PIL.Image.fromarray(image)

        # Apply the transformation
        output = image_transform({"image": image})
        output_image = output["image"]
        output_ar = output["aspect_ratio"]

        inference_output = image_transform_inference(image=image)
        inference_output_image = inference_output["image"]
        inference_output_ar = inference_output["aspect_ratio"]

        # Check output is the same across CLIPImageTransform and CLIPImageTransformInference.
        assert torch.allclose(output_image, inference_output_image)
        assert output_ar[0] == inference_output_ar[0]
        assert output_ar[1] == inference_output_ar[1]

        # Check if the output shape matches the expected shape
        assert (
            output_image.shape == params["expected_shape"]
        ), f"Expected shape {params['expected_shape']} but got {output_image.shape}"

        # Check if the pixel values are within the expected range [0, 1]
        assert (
            0 <= output_image.min() <= output_image.max() <= 1
        ), f"Expected pixel values to be in range [0, 1] but got {output_image.min()} and {output_image.max()}"

        # Check if the mean, max, and min values of the tiles match the expected values
        for i, tile in enumerate(output_image):
            assert_expected(
                tile.mean().item(), params["expected_tile_means"][i], rtol=0, atol=1e-4
            )
            assert_expected(
                tile.max().item(), params["expected_tile_max"][i], rtol=0, atol=1e-4
            )
            assert_expected(
                tile.min().item(), params["expected_tile_min"][i], rtol=0, atol=1e-4
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
