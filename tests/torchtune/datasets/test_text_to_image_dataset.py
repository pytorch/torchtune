# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
from PIL import Image

from torchtune.datasets import text_to_image_dataset
from torchvision import transforms


class TestTextToImageDataset:
    @pytest.fixture
    def img_dir(self, tmp_path):
        # Create directory with two small images (one red and one green)
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        Image.new("RGB", (16, 16), (255, 0, 0)).save(img_dir / "0.png")
        Image.new("RGB", (16, 16), (0, 255, 0)).save(img_dir / "1.png")
        return img_dir

    def test_text_to_image_dataset(self, tmp_path, img_dir):
        # Create dataset json file
        ds_path = tmp_path / "ds.json"
        with open(ds_path, "w") as f:
            json.dump(
                [
                    {"text": "caption0", "image": "0.png"},
                    {"text": "caption1", "image": "1.png"},
                ],
                f,
            )

        # Create dataset
        ds = text_to_image_dataset(
            _DummyTransform(),
            source="json",
            data_files=str(ds_path),
            image_dir=str(img_dir),
        )

        assert len(ds) == 2

        # Check first row of data
        row = ds[0]
        assert len(row) == 2
        assert "".join(chr(x) for x in row["text"]) == "caption0"
        assert row["image"].shape == (3, 16, 16)
        assert row["image"].mean(dim=(1, 2)).numpy().tolist() == [1.0, 0.0, 0.0]

        # Check second row of data
        row = ds[1]
        assert len(row) == 2
        assert "".join(chr(x) for x in row["text"]) == "caption1"
        assert row["image"].shape == (3, 16, 16)
        assert row["image"].mean(dim=(1, 2)).numpy().tolist() == [0.0, 1.0, 0.0]

    def test_include_id(self, tmp_path, img_dir):
        # Create dataset json file
        ds_path = tmp_path / "ds.json"
        with open(ds_path, "w") as f:
            json.dump(
                [
                    {"id": "0", "text": "caption0", "image": "0.png"},
                    {"id": "1", "text": "caption1", "image": "1.png"},
                ],
                f,
            )

        # Create dataset
        ds = text_to_image_dataset(
            _DummyTransform(),
            source="json",
            data_files=str(ds_path),
            image_dir=str(img_dir),
            include_id=True,
        )

        assert len(ds) == 2
        assert len(ds[0]) == 3
        assert ds[0]["id"] == "0"
        assert ds[1]["id"] == "1"

    def test_column_map(self, tmp_path, img_dir):
        # Create dataset json file
        ds_path = tmp_path / "ds.json"
        with open(ds_path, "w") as f:
            json.dump(
                [
                    {"text_": "caption0", "image_": "0.png"},
                    {"text_": "caption1", "image_": "1.png"},
                ],
                f,
            )

        # Create dataset
        ds = text_to_image_dataset(
            _DummyTransform(),
            source="json",
            data_files=str(ds_path),
            image_dir=str(img_dir),
            column_map={"text": "text_", "image": "image_"},
        )

        assert len(ds) == 2
        assert len(ds[0]) == 2
        assert "text" in ds[0]
        assert "image" in ds[0]


class _DummyTransform:
    def __init__(self):
        self._img_transform = transforms.ToTensor()

    def __call__(self, sample):
        return {
            "image": self._img_transform(sample["image"]),
            "text": [ord(x) for x in sample["text"]],
        }
