# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from PIL import Image

from tests.common import ASSETS
from torchtune.models.flux import FluxTransform


class TestFluxTransform:
    @pytest.fixture
    def img(self):
        return Image.new("RGB", (16, 16), (255, 0, 0))

    def test_flux_transform(self, img):
        # Create transform
        transform = FluxTransform(
            flux_model_name="FLUX.1-dev",
            img_width=8,
            img_height=8,
            clip_tokenizer_path=str(ASSETS / "tiny_bpe_merges.txt"),
            t5_tokenizer_path=str(ASSETS / "sentencepiece.model"),
        )

        # Tranform a row of data
        x = transform({"image": img, "text": "test"})

        assert len(x) == 3  # image, clip tokens, and t5 tokens

        # Check that image was cropped and normalized
        assert x["image"].shape == (3, 8, 8)
        assert torch.min(x["image"]).item() == -1.0
        assert torch.max(x["image"]).item() == 1.0

        # Check that the tokens are the correct sequence length
        assert x["clip_tokens"].shape == (77,)
        assert x["t5_tokens"].shape == (512,)

    def test_truncation(self, img):
        data = {"image": img, "text": "x" * 10_000}

        # Check that long sequences raise an error if truncate_text is false
        transform = FluxTransform(
            flux_model_name="FLUX.1-dev",
            img_width=8,
            img_height=8,
            clip_tokenizer_path=str(ASSETS / "tiny_bpe_merges.txt"),
            t5_tokenizer_path=str(ASSETS / "sentencepiece.model"),
            truncate_text=False,
        )
        with pytest.raises(AssertionError):
            x = transform(data)

        # Check that long sequences are truncated if truncate_text is true
        transform = FluxTransform(
            flux_model_name="FLUX.1-dev",
            img_width=8,
            img_height=8,
            clip_tokenizer_path=str(ASSETS / "tiny_bpe_merges.txt"),
            t5_tokenizer_path=str(ASSETS / "sentencepiece.model"),
            truncate_text=True,
        )
        x = transform(data)
        assert x["clip_tokens"].shape == (77,)
        assert x["t5_tokens"].shape == (512,)

    def test_flux_schnell_transform(self, img):
        # FLUX.1-schnell has a smaller T5 max seq len
        transform = FluxTransform(
            flux_model_name="FLUX.1-schnell",
            img_width=8,
            img_height=8,
            clip_tokenizer_path=str(ASSETS / "tiny_bpe_merges.txt"),
            t5_tokenizer_path=str(ASSETS / "sentencepiece.model"),
        )
        x = transform({"image": img, "text": "test"})
        assert x["t5_tokens"].shape == (256,)
