# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import assert_expected, fixed_init_model, fixed_init_tensor
from torchtune.models.llama3_2_vision._component_builders import llama3_2_vision_encoder


@pytest.fixture
def transformer_config():
    return {
        "clip_embed_dim": 32,
        "clip_num_layers": 2,
        "num_heads": 4,
        "tile_size": 49,
        "patch_size": 9,
        "max_num_tiles": 4,
        "in_channels": 3,
        "clip_hidden_states": [0, 1],
        "num_layers_projection": 2,
        "decoder_embed_dim": 128,
    }


@pytest.fixture
def vision_transformer(transformer_config):
    vision_transformer = llama3_2_vision_encoder(**transformer_config).eval()
    fixed_init_model(vision_transformer, min_val=-1, max_val=1)
    return vision_transformer


class TestLlama3VisionEncoder:
    @pytest.fixture(autouse=True)
    def setup_class(self, transformer_config):
        self.batch_size = 1
        self.n_imgs = 2
        self.num_tiles = 4
        self.aspect_ratio = torch.tensor([[1, 3], [2, 2]]).reshape(
            self.batch_size, self.n_imgs, 2
        )
        image = torch.rand(
            (
                self.batch_size,
                self.n_imgs,
                self.num_tiles,
                transformer_config["in_channels"],
                transformer_config["tile_size"],
                transformer_config["tile_size"],
            )
        )
        self.image = fixed_init_tensor(image.shape, min_val=-1, max_val=1)

    def test_llama3_2_vision_encoder(self, vision_transformer, transformer_config):
        # call model
        output = vision_transformer(self.image, self.aspect_ratio)

        # assertion
        expected_shape = (
            self.batch_size,
            self.n_imgs
            * self.num_tiles
            * vision_transformer.clip.get_image_tokens_per_tile(),
            transformer_config["decoder_embed_dim"],
        )
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(5.28800), atol=1e-3, rtol=1e-3)

    def test_fails_if_ar_none_and_multiple_tiles(self, vision_transformer):
        assert self.image.shape[2] > 1, "This test is not valid for num_tiles=1"
        try:
            vision_transformer(self.image, aspect_ratio=None)
            pytest.fail(
                "Expected ValueError: If num_tiles>1, aspect_ratio should not be None"
            )
        except ValueError:
            pass  # If ValueError is raised, the test passes

    def test_llama3_2_vision_encoder_single_tile(self, transformer_config):
        transformer_config = transformer_config.copy()
        transformer_config["max_num_tiles"] = 1
        images = self.image[:, :, [0], :, :, :]

        model_with_multiple_tiles = llama3_2_vision_encoder(**transformer_config).eval()
        fixed_init_model(model_with_multiple_tiles, min_val=-1, max_val=1)

        output = model_with_multiple_tiles(images, aspect_ratio=None)

        expected_shape = (
            self.batch_size,
            self.n_imgs * model_with_multiple_tiles.clip.get_image_tokens_per_tile(),
            transformer_config["decoder_embed_dim"],
        )
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(5.38046), atol=1e-3, rtol=1e-3)

    def test_llama3_2_no_hidden_layers(self, vision_transformer, transformer_config):
        # Modify the transformer_config for this specific test
        transformer_config = transformer_config.copy()
        transformer_config["clip_hidden_states"] = None

        # Reinitialize the model with the updated configuration
        model_with_no_hidden = llama3_2_vision_encoder(**transformer_config).eval()
        fixed_init_model(model_with_no_hidden, min_val=-1, max_val=1)

        # Call model
        output = model_with_no_hidden(self.image, self.aspect_ratio)

        # Assertion
        expected_shape = (
            self.batch_size,
            self.n_imgs
            * self.num_tiles
            * model_with_no_hidden.clip.get_image_tokens_per_tile(),
            transformer_config["decoder_embed_dim"],
        )
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(-77.3419), atol=1e-3, rtol=1e-3)
