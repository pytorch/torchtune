# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.test_utils import assert_expected
from torchtune.models.clip._component_builders import clip

from torchtune.utils.seed import set_seed


@pytest.fixture
def random(auto_use=True):
    set_seed(42)


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
    set_seed(42)
    return clip(**transformer_config).eval()


class TestVisionTransformer:
    @pytest.fixture(autouse=True)
    def setup_class(self, transformer_config):
        self.batch_size = 2
        num_channels = transformer_config["in_channels"]

        # generate random aspect ratios up to max_num_tiles
        set_seed(42)
        aspect_ratio = []
        curr_max_num_tiles = 0
        while len(aspect_ratio) < self.batch_size:
            aspect_ratio_candidate = torch.randint(
                1, transformer_config["max_num_tiles"], (2,)
            )
            num_tiles = int(torch.prod(aspect_ratio_candidate))
            if num_tiles <= transformer_config["max_num_tiles"]:
                aspect_ratio.append(aspect_ratio_candidate)
                curr_max_num_tiles = max(num_tiles, curr_max_num_tiles)

        self.aspect_ratio = torch.stack(aspect_ratio, dim=0)
        self.num_tiles = curr_max_num_tiles

        # generate random image
        set_seed(42)
        self.image = torch.rand(
            (
                self.batch_size,
                self.num_tiles,
                num_channels,
                transformer_config["tile_size"],
                transformer_config["tile_size"],
            )
        )

    def test_vision_transformer_output_shape(
        self, vision_transformer, transformer_config
    ):
        # call model
        output = vision_transformer(self.image, self.aspect_ratio)

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

    def test_vision_transformer_with_cls_projection(self, transformer_config):
        transformer_config = transformer_config.copy()
        transformer_config["output_cls_projection"] = True

        # call model
        set_seed(42)
        model_with_cls = clip(**transformer_config).eval()
        output = model_with_cls(self.image, None)

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

    def test_vision_transformer_return_hidden_layers(self, transformer_config):
        transformer_config = transformer_config.copy()
        transformer_config["indices_return_hidden"] = [
            0,
            1,
        ]

        # call model
        set_seed(42)
        model_with_hidden = clip(**transformer_config)
        x, hidden_layers = model_with_hidden(self.image)

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

        assert_expected(x.mean(), torch.tensor(2.2925e-08), atol=1e-3, rtol=1e-3)

        # assertion hidden
        num_hidden_layers_expected = len(transformer_config["indices_return_hidden"])

        expected_shape_hidden_layers = (
            self.batch_size,
            num_hidden_layers_expected,
            self.num_tiles,
            model_with_hidden.get_image_tokens_per_tile(),
            transformer_config["embed_dim"],
        )
        assert (
            hidden_layers.shape == expected_shape_hidden_layers
        ), f"Expected shape {expected_shape_hidden_layers}, but got {hidden_layers.shape=}"

        assert_expected(
            hidden_layers.mean(), torch.tensor(-0.0309), atol=1e-3, rtol=1e-3
        )

    def test_vision_transformer_single_tile(self, transformer_config):
        transformer_config = transformer_config.copy()
        images = self.image[:, 0, :, :]

        # call model
        model_with_multiple_tiles = clip(**transformer_config)
        output = model_with_multiple_tiles(images, aspect_ratio=None)

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
