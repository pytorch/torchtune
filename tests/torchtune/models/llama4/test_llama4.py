# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import fixed_init_model
from torchtune.models.llama4 import llama4_vision_encoder, llama4_vision_projection_head
from torchtune.training.seed import set_seed

EMBED_DIM = 128
BSZ = 2
N_IMG = 1
N_TILES = 4
TILE_SIZE = 64
PATCH_SIZE = 16
N_PATCHES = (TILE_SIZE // PATCH_SIZE) ** 2 + 1  # 16 + 1 for CLS token
N_TIME_STEPS = 256
N_HEADS = 16


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestLlama4:
    @pytest.fixture
    def patch_inputs(self):
        return torch.randn((BSZ * N_IMG * N_TILES, N_PATCHES, EMBED_DIM))

    @pytest.fixture
    def image_inputs(self):
        return torch.randn((BSZ * N_IMG * N_TILES, 3, TILE_SIZE, TILE_SIZE))

    def test_vision_projection_head_forward(self, patch_inputs):
        """
        Verified against an internal implementation of the vision projection head.
        """
        head = llama4_vision_projection_head(
            clip_embed_dim=EMBED_DIM,
            decoder_embed_dim=EMBED_DIM,
            projection_embed_dim=EMBED_DIM * 2,
        )
        fixed_init_model(head, min_val=-0.25, max_val=0.5)
        actual = head(patch_inputs)
        expected = torch.tensor(2463.2561)
        # The CLS patch is dropped, and divide by 4 to account for pixel shuffle
        assert actual.shape == (BSZ * N_IMG * N_TILES, (N_PATCHES - 1) // 4, EMBED_DIM)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-4)

    def test_vision_encoder_forward(self, image_inputs):
        """
        Verified against an internal implementation of the vision encoder.
        """
        encoder = llama4_vision_encoder(
            patch_size=PATCH_SIZE,
            num_heads=N_HEADS,
            clip_embed_dim=EMBED_DIM,
            clip_num_layers=2,
            projection_embed_dim=EMBED_DIM * 2,
            decoder_embed_dim=EMBED_DIM,
            tile_size=TILE_SIZE,
            max_num_tiles=N_TILES,
            in_channels=3,
        )
        fixed_init_model(encoder, min_val=-0.25, max_val=0.5)
        actual = encoder(image_inputs)
        expected = torch.tensor(40112.8438)
        # The CLS patch is dropped, and divide by 4 to account for pixel shuffle
        assert actual.shape == (BSZ * N_IMG * N_TILES, (N_PATCHES - 1) // 4, EMBED_DIM)
        torch.testing.assert_close(actual.mean(), expected, atol=1e-4, rtol=1e-4)
