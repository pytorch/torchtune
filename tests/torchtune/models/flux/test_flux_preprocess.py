# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import pytest
import torch
from PIL import Image

from torchtune.datasets import text_to_image_dataset
from torchtune.models.flux._preprocess import FluxPreprocessor
from torchvision import transforms


class TestFluxPreprocessor:
    @pytest.fixture
    def preprocessor(self, tmp_path):
        return FluxPreprocessor(
            autoencoder=_DummyAutoencoder(),
            clip_encoder=_DummyClipEnc(),
            t5_encoder=_DummyT5Enc(),
            preprocessed_data_dir=tmp_path / "preprocessed",
            preprocess_again_if_exists=True,
            batch_size=1,
            device="cpu",
            dtype=torch.float32,
        )

    def test_preprocess_ds(self, tmp_path, preprocessor):
        # Create directory with two small images
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        Image.new("RGB", (16, 16), (255, 0, 0)).save(img_dir / "0.png")
        Image.new("RGB", (16, 16), (0, 255, 0)).save(img_dir / "1.png")

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

        # Preprocess dataset
        preprocess_ds = preprocessor.preprocess_dataset(ds)

        assert len(preprocess_ds) == 2

        row = preprocess_ds[0]
        assert len(row) == 3
        assert row["img_encoding"].shape == (8, 2, 2)
        assert row["clip_text_encoding"].shape == (8,)
        assert row["t5_text_encoding"].shape == (256, 8)

    def test_preprocess_text(self, preprocessor):
        clip_encs, t5_encs = preprocessor.preprocess_text(
            ["a", "b", "c"], _dummy_tokenize_fn
        )
        assert clip_encs.shape == (3, 8)
        assert t5_encs.shape == (3, 256, 8)


class _DummyAutoencoder(torch.nn.Module):
    def encode(self, img):
        assert img.shape == (1, 3, 16, 16)
        return torch.randn(1, 8, 2, 2)


class _DummyClipEnc(torch.nn.Module):
    def forward(self, tokens):
        assert tokens.shape == (1, 77)
        return torch.randn(1, 8)


class _DummyT5Enc(torch.nn.Module):
    def forward(self, tokens):
        assert tokens.shape == (1, 256)
        return torch.randn(1, 256, 8)


def _dummy_tokenize_fn(text):
    assert isinstance(text, str)
    clip_tokens = torch.randn(77)
    t5_tokens = torch.randn(256)
    return clip_tokens, t5_tokens


class _DummyTransform:
    def __init__(self):
        self._img_transform = transforms.ToTensor()

    def __call__(self, sample):
        return {
            "image": self._img_transform(sample["image"]),
            "clip_tokens": torch.randn(77),
            "t5_tokens": torch.randn(256),
        }
