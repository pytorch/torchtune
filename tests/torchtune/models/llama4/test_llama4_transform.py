# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.common import ASSETS
from tests.test_utils import assert_expected
from torchtune.models.llama4._transform import Llama4Transform
from torchtune.training.seed import set_seed
from torchvision.transforms.v2 import functional as F

EMBED_DIM = 128
BSZ = 2
N_IMG = 1
N_TILES = 4
N_PATCHES = 17  # 16 + 1 for CLS token


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


def random_image():
    tensor = torch.randn(3, 4, 4)
    tensor = torch.clamp(tensor, 0, 1)
    pil_image = F.to_pil_image(tensor)
    return pil_image


class TestImageThumbnailTransform:
    @pytest.fixture
    def transform(self):
        return Llama4Transform(
            path=str(ASSETS / "tiktoken_small_llama4.model"),
            tile_size=16,
            patch_size=4,
            max_num_tiles=4,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            dtype=torch.float32,
        )

    def test_call(self, transform):
        image = random_image()
        actual = transform.thumbnail_transform(image)
        assert actual.shape == (3, 16, 16)
        assert_expected(actual.sum(), torch.tensor(-123.3412))
