# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected, fixed_init_model, fixed_init_tensor
from torchtune.models.clip._component_builders import clip_vision_encoder


@pytest.fixture
def transformer_config():
    return {
        "embed_dim": 32,
        "cls_output_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "tile_size": 49,
        "patch_size": 9,
        "max_num_tiles": 4,
        "mlp_ratio": 4.0,
        "act_layer": torch.nn.SiLU(),
        "in_channels": 3,
        "attn_dropout": 0.0,
        "norm_eps": 1e-5,
        "output_cls_projection": False,
        "indices_return_hidden": None,
    }


@pytest.fixture
def vision_transformer(transformer_config):
    vision_transformer = clip_vision_encoder(**transformer_config).eval()
    fixed_init_model(vision_transformer, min_val=-1, max_val=1)
    return vision_transformer


class TestVisionTransformer:
    @pytest.fixture(autouse=True)
    def setup_class(self, transformer_config):
        self.batch_size = 2
        num_channels = transformer_config["in_channels"]

        # generate aspect ratios up to max_num_tiles
        self.aspect_ratio = torch.tensor([[1, 3], [2, 2]])
        self.num_tiles = 4
        assert (
            self.num_tiles <= transformer_config["max_num_tiles"]
        ), "For this test to be valid, num_tiles should be <= max_num_tiles"
        assert (
            torch.prod(self.aspect_ratio, dim=-1).max() <= self.num_tiles
        ), "For this test to be vlaid, prod(aspect_ratio).max() should match num_tiles"

        # generate image
        image = torch.rand(
            (
                self.batch_size,
                self.num_tiles,
                num_channels,
                transformer_config["tile_size"],
                transformer_config["tile_size"],
            )
        )
        self.image = fixed_init_tensor(image.shape, min_val=-1, max_val=1)

    def test_vision_transformer_without_hidden_layers(
        self, vision_transformer, transformer_config
    ):
        # call model
        output, _ = vision_transformer(self.image, self.aspect_ratio)

        # assertion
        expected_shape = (
            self.batch_size,
            self.num_tiles,
            vision_transformer.get_image_tokens_per_tile(),
            transformer_config["embed_dim"],
        )
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(1.0172), atol=1e-3, rtol=1e-3)

    def test_fails_if_ar_none_and_multiple_tiles(self, vision_transformer):
        assert self.image.shape[1] > 1, "This test is not valid for num_tiles=1"
        try:
            vision_transformer(self.image, aspect_ratio=None)
            pytest.fail(
                "Expected ValueError: If num_tiles>1, aspect_ratio should not be None"
            )
        except ValueError:
            pass  # If ValueError is raised, the test passes

    def test_vision_transformer_with_cls_projection(self, transformer_config):
        transformer_config = transformer_config.copy()
        transformer_config["output_cls_projection"] = True

        # call model
        model_with_cls = clip_vision_encoder(**transformer_config).eval()
        fixed_init_model(model_with_cls, min_val=-1, max_val=1)
        output, _ = model_with_cls(self.image, self.aspect_ratio)

        # assertion
        expected_shape = (
            self.batch_size,
            self.num_tiles,
            1,
            transformer_config["cls_output_dim"],
        )

        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(9.6240), atol=1e-3, rtol=1e-3)

    def test_vision_transformer_return_hidden_layers(self, transformer_config):
        transformer_config = transformer_config.copy()
        transformer_config["indices_return_hidden"] = [
            0,
            1,
        ]

        # call model
        model_with_hidden = clip_vision_encoder(**transformer_config)
        fixed_init_model(model_with_hidden, min_val=-1, max_val=1)
        x, hidden_layers = model_with_hidden(self.image, self.aspect_ratio)

        # assertion x
        expected_shape_x = (
            self.batch_size,
            self.num_tiles,
            model_with_hidden.get_image_tokens_per_tile(),
            transformer_config["embed_dim"],
        )

        assert (
            x.shape == expected_shape_x
        ), f"Expected shape {expected_shape_x}, but got {x.shape=}"

        assert_expected(x.mean(), torch.tensor(1.0172), atol=1e-3, rtol=1e-3)

        # assertion hidden
        num_hidden_layers_expected = len(transformer_config["indices_return_hidden"])

        expected_shape_hidden_layers = (
            self.batch_size,
            self.num_tiles,
            model_with_hidden.get_image_tokens_per_tile(),
            transformer_config["embed_dim"],
            num_hidden_layers_expected,
        )
        assert (
            hidden_layers.shape == expected_shape_hidden_layers
        ), f"Expected shape {expected_shape_hidden_layers}, but got {hidden_layers.shape=}"

        assert_expected(
            hidden_layers.mean(), torch.tensor(6.6938), atol=1e-3, rtol=1e-3
        )

    def test_vision_transformer_single_tile(self, transformer_config):
        transformer_config = transformer_config.copy()
        transformer_config["max_num_tiles"] = 1
        images = self.image[:, [0], :, :, :]

        # call model
        model_with_multiple_tiles = clip_vision_encoder(**transformer_config)
        fixed_init_model(model_with_multiple_tiles, min_val=-1, max_val=1)
        output, _ = model_with_multiple_tiles(images, aspect_ratio=None)

        # assertion
        expected_shape = (
            self.batch_size,
            1,
            model_with_multiple_tiles.get_image_tokens_per_tile(),
            transformer_config["embed_dim"],
        )
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        assert_expected(output.mean(), torch.tensor(0.5458), atol=1e-3, rtol=1e-3)
