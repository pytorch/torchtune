# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.modules.transforms import VisionCrossAttentionMask


IMAGE_TOKEN_ID = 1
MAX_NUM_TILES = 4


class TestVisionCrossAttentionMask:
    @pytest.fixture
    def num_tiles(self):
        return 2

    @pytest.fixture
    def tile_size(self):
        return 4

    @pytest.fixture
    def patch_size(self):
        return 2

    @pytest.fixture
    def image_num_tokens(self, num_tiles, tile_size, patch_size):
        return ((tile_size // patch_size) ** 2 + 1) * num_tiles

    @pytest.fixture
    def tokens(self):
        # This tests image tokens not at start, consecutive images, and image
        # with text until end.
        # text = 2, image = 1
        return [2, 2, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2, 2, IMAGE_TOKEN_ID, 2, 2]

    @pytest.fixture
    def images(self, num_tiles, tokens):
        n_img = len([token_id for token_id in tokens if token_id == IMAGE_TOKEN_ID])
        n_channels = 3
        tile_size = 2
        return [
            torch.ones(num_tiles, n_channels, tile_size, tile_size)
            for _ in range(n_img)
        ]

    @pytest.fixture
    def cross_attn_mask_transform(self, tile_size, patch_size):
        # patches per tile = 4
        return VisionCrossAttentionMask(
            tile_size=tile_size,
            patch_size=patch_size,
            image_token_id=IMAGE_TOKEN_ID,
            max_num_tiles=MAX_NUM_TILES,
        )

    def test_get_image_attention_intervals(self, cross_attn_mask_transform, tokens):
        actual = cross_attn_mask_transform._get_image_attention_intervals(tokens)
        expected = [[2, 6], [3, 6], [6, 9]]
        assert actual == expected

    def test_call(self, cross_attn_mask_transform, tokens, images, image_num_tokens):
        sample = {"tokens": tokens, "encoder_input": {"images": images}}
        dummy_kwargs = {"hello": 8}
        sample.update(dummy_kwargs)
        actual = cross_attn_mask_transform(sample)
        expected = [
            torch.zeros(len(tokens), image_num_tokens, dtype=torch.bool)
            for _ in range(len(images))
        ]
        expected[0][2:6, :] = True
        expected[1][3:6, :] = True
        expected[2][6:9, :] = True
        for i in range(len(images)):
            torch.testing.assert_close(actual["encoder_mask"][i], expected[i])
            torch.testing.assert_close(actual["encoder_input"]["images"][i], images[i])

        assert actual["tokens"] == tokens
        assert actual["hello"] == dummy_kwargs["hello"]

    def test_inference_call(
        self, cross_attn_mask_transform, tokens, images, image_num_tokens
    ):
        sample = {"tokens": tokens, "encoder_input": {"images": images}}
        dummy_kwargs = {"hello": 8}
        sample.update(dummy_kwargs)
        actual = cross_attn_mask_transform(sample, inference=True)
        expected = [
            torch.zeros(len(tokens), image_num_tokens * 2, dtype=torch.bool)
            for _ in range(len(images))
        ]
        expected[0][2:6, :image_num_tokens] = True
        expected[1][3:6, :image_num_tokens] = True
        expected[2][6:9, :image_num_tokens] = True
        for i in range(len(images)):
            torch.testing.assert_close(actual["encoder_mask"][i], expected[i])
            torch.testing.assert_close(actual["encoder_input"]["images"][i], images[i])

        assert actual["tokens"] == tokens
        assert actual["hello"] == dummy_kwargs["hello"]
