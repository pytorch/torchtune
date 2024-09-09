# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtune.models.flamingo._transform import _load_image, FlamingoTransform


class TestFlamingoTransform:
    """Test the FlamingoTransform class."""

    def test_load_image_https(self, tmpdir, mocker):
        # Mock a response from a URL
        mock_response = mocker.Mock()
        mock_response.status = 200
        # Mock response data as a binary stream
        mock_response.data = b"image data"
        mocker.patch("urllib.requests.urlopen", return_value=mock_response)

        # Test that we can load an image from a URL
        image = _load_image("https://loremflickr.com/640/480/dog")
        assert image.shape == (3, 480, 640)

    def test_load_image_locally(self, tmpdir):
        # Test that we can load an image from a local file
        image_path = os.path.join(tmpdir, "test_image.jpg")
        image = Image.new("RGB", (640, 480), color="red")
        image.save(image_path)
        image = _load_image(image_path)
        assert image.shape == (3, 480, 640)

    def test_load_image_invalid(self, tmpdir):
        # Test that we get an error when trying to load an invalid image
        with pytest.raises(ValueError):
            _load_image("invalid_image_path.jpg")

        with pytest.raises(ValueError):
            _load_image("https://invalid_image_url.com")
