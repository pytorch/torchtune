# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from tests.test_utils import assert_expected

from torchtune.models.clip._position_embeddings import (
    TiledTokenPositionalEmbedding,
    TilePositionalEmbedding,
)

# generated comparing vs fairinternal/internal-llama-models
tile_pos_emb_test_cases = [
    {
        "tgt_max_num_tiles": 1,
        "input_tensor": torch.tensor(
            [[[[0.0, 1.0]], [[2.0, 3.0]]], [[[4.0, 5.0]], [[6.0, 7.0]]]]
        ),
        "expected_output": torch.tensor([[[[0.0, 1.0]]]]),
    },
    {
        "tgt_max_num_tiles": 3,
        "input_tensor": torch.tensor([[[[0.0]]]]),
        "expected_output": torch.tensor(
            [
                [[[0.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]], [[0.0]]],
                [[[0.0]], [[0.0]], [[0.0]]],
            ]
        ),
    },
    {
        "tgt_max_num_tiles": 2,
        "input_tensor": torch.tensor(
            [
                [[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]],
                [[[6.0, 7.0]], [[8.0, 9.0]], [[10.0, 11.0]]],
                [[[12.0, 13.0]], [[14.0, 15.0]], [[16.0, 17.0]]],
            ]
        ),
        "expected_output": torch.tensor(
            [[[[0.0, 1.0]], [[4.0, 5.0]]], [[[12.0, 13.0]], [[16.0, 17.0]]]]
        ),
    },
]

local_pos_emb_test_cases = [
    {
        "tgt_patch_grid_size": 2,
        "expected_shape": torch.Size([5, 2]),
        "input_tensor": torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
        ),
        "expected_output": torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
        ),
    },
    {
        "tgt_patch_grid_size": 1,
        "expected_shape": torch.Size([2, 1]),
        "input_tensor": torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        "expected_output": torch.tensor([[0.0], [1.0]]),
    },
    {
        "tgt_patch_grid_size": 2,
        "expected_shape": torch.Size([5, 2]),
        "input_tensor": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "expected_output": torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]
        ),
    },
]

global_pos_emb_test_cases = [
    {
        "tgt_max_num_tiles": 1,
        "tgt_patch_grid_size": 2,
        "input_tensor": torch.tensor(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
                    [
                        [10.0, 11.0],
                        [12.0, 13.0],
                        [14.0, 15.0],
                        [16.0, 17.0],
                        [18.0, 19.0],
                    ],
                ],
                [
                    [
                        [20.0, 21.0],
                        [22.0, 23.0],
                        [24.0, 25.0],
                        [26.0, 27.0],
                        [28.0, 29.0],
                    ],
                    [
                        [30.0, 31.0],
                        [32.0, 33.0],
                        [34.0, 35.0],
                        [36.0, 37.0],
                        [38.0, 39.0],
                    ],
                ],
            ]
        ),
        "expected_output": torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0], [14.0, 15.0], [26.0, 27.0], [38.0, 39.0]]]]
        ),
    },
    {
        "tgt_max_num_tiles": 3,
        "tgt_patch_grid_size": 1,
        "input_tensor": torch.tensor([[[[0.0], [1.0], [2.0], [3.0], [4.0]]]]),
        "expected_output": torch.tensor(
            [
                [[[0.0000], [1.0000]], [[0.0000], [1.5000]], [[0.0000], [2.0000]]],
                [[[0.0000], [2.0000]], [[0.0000], [2.5000]], [[0.0000], [3.0000]]],
                [[[0.0000], [3.0000]], [[0.0000], [3.5000]], [[0.0000], [4.0000]]],
            ]
        ),
    },
    {
        "tgt_max_num_tiles": 2,
        "tgt_patch_grid_size": 2,
        "input_tensor": torch.tensor(
            [
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                    [[8.0, 9.0], [10.0, 11.0]],
                ],
                [
                    [[12.0, 13.0], [14.0, 15.0]],
                    [[16.0, 17.0], [18.0, 19.0]],
                    [[20.0, 21.0], [22.0, 23.0]],
                ],
                [
                    [[24.0, 25.0], [26.0, 27.0]],
                    [[28.0, 29.0], [30.0, 31.0]],
                    [[32.0, 33.0], [34.0, 35.0]],
                ],
            ]
        ),
        "expected_output": torch.tensor(
            [
                [
                    [
                        [0.0000, 1.0000],
                        [2.0000, 3.0000],
                        [4.6667, 5.6667],
                        [10.0000, 11.0000],
                        [12.6667, 13.6667],
                    ],
                    [
                        [8.0000, 9.0000],
                        [7.3333, 8.3333],
                        [10.0000, 11.0000],
                        [15.3333, 16.3333],
                        [18.0000, 19.0000],
                    ],
                ],
                [
                    [
                        [24.0000, 25.0000],
                        [18.0000, 19.0000],
                        [20.6667, 21.6667],
                        [26.0000, 27.0000],
                        [28.6667, 29.6667],
                    ],
                    [
                        [32.0000, 33.0000],
                        [23.3333, 24.3333],
                        [26.0000, 27.0000],
                        [31.3333, 32.3333],
                        [34.0000, 35.0000],
                    ],
                ],
            ]
        ),
    },
]


class TestPositionalEmbeddingsInterpolation:
    @pytest.mark.parametrize("params", tile_pos_emb_test_cases)
    def test_tile_resize_position_embedding(self, params):
        tgt_max_num_tiles = params["tgt_max_num_tiles"]
        expected_output = params["expected_output"]
        embedding = params["input_tensor"]

        resized_pos_embed = TilePositionalEmbedding._resize_position_embedding(
            embedding, tgt_max_num_tiles
        )

        assert_expected(resized_pos_embed, expected_output, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize("params", local_pos_emb_test_cases)
    def test_resize_local_position_embedding(self, params):
        input_tensor = params["input_tensor"]
        tgt_patch_grid_size = params["tgt_patch_grid_size"]
        expected_output = params["expected_output"]

        resized_pos_embed = (
            TiledTokenPositionalEmbedding._resize_local_position_embedding(
                input_tensor, tgt_patch_grid_size
            )
        )

        assert_expected(resized_pos_embed, expected_output, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize("params", global_pos_emb_test_cases)
    def test_resize_global_position_embedding(self, params):
        input_tensor = params["input_tensor"]
        tgt_max_num_tiles = params["tgt_max_num_tiles"]
        tgt_patch_grid_size = params["tgt_patch_grid_size"]
        expected_output = params["expected_output"]

        resized_pos_embed = (
            TiledTokenPositionalEmbedding._resize_global_position_embedding(
                input_tensor, tgt_max_num_tiles, tgt_patch_grid_size
            )
        )

        assert_expected(resized_pos_embed, expected_output, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize(
        "local_params, global_params",
        zip(local_pos_emb_test_cases, global_pos_emb_test_cases),
    )
    def test_load_state_dict_hook_tiled_token(self, local_params, global_params):
        # Corrected parameters for instantiation
        global_max_num_tiles = global_params["expected_output"].shape[0]
        global_embed_dim = global_params["expected_output"].shape[-1]
        n_tokens_per_tile = local_params["expected_output"].shape[
            0
        ]  # Assuming first dimension is tokens per tile
        patch_grid_size = int(math.sqrt(n_tokens_per_tile - 1))
        tile_size = patch_grid_size * 1  # Assuming patch_size is 1 for simplicity
        patch_size = 1

        # Instantiate the model
        model = TiledTokenPositionalEmbedding(
            max_num_tiles=global_max_num_tiles,
            embed_dim=global_embed_dim,
            tile_size=tile_size,
            patch_size=patch_size,
        )

        # Create state_dict mimicking loading scenario
        state_dict = {
            "model.local_token_positional_embedding": local_params["input_tensor"],
            "model.global_token_positional_embedding": global_params["input_tensor"],
        }

        # Call the hook directly (simulating loading process)
        state_dict_copy = state_dict.copy()
        model._load_state_dict_hook(state_dict_copy, "model.")

        # Assert expected outputs
        assert_expected(
            state_dict_copy["model.local_token_positional_embedding"],
            local_params["expected_output"],
            atol=1e-3,
            rtol=1e-4,
        )
        assert_expected(
            state_dict_copy["model.global_token_positional_embedding"],
            global_params["expected_output"],
            atol=1e-3,
            rtol=1e-4,
        )

        # Check for errors with non-square token grid sizes
        with pytest.raises(ValueError):
            bad_state_dict = state_dict.copy()

            # add +1 to num_token dimension to make it non-square
            local_pos_emb = bad_state_dict["model.local_token_positional_embedding"]
            bad_local_pos_emb = torch.cat(
                (local_pos_emb, local_pos_emb[0].unsqueeze(0)), dim=0
            )
            bad_state_dict["model.local_token_positional_embedding"] = bad_local_pos_emb

            # call
            model._load_state_dict_hook(bad_state_dict, "model.")

        # Check for errors with non-square tile grid sizes
        with pytest.raises(ValueError):
            bad_state_dict = state_dict.copy()

            # add +1 to num_token dimension to make it non-square
            global_pos_emb = bad_state_dict["model.global_token_positional_embedding"]
            bad_global_pos_emb = torch.cat(
                (global_pos_emb, global_pos_emb[:, :, [0]]), dim=2
            )
            bad_state_dict[
                "model.global_token_positional_embedding"
            ] = bad_global_pos_emb

            # call
            model._load_state_dict_hook(bad_state_dict, "model.")

    @pytest.mark.parametrize("params", tile_pos_emb_test_cases)
    def test_load_state_dict_hook_tile(self, params):

        # Extract parameters for instantiation
        max_num_tiles = params["expected_output"].shape[0]
        embed_dim = params["expected_output"].shape[-1]

        # Instantiate the model
        model = TilePositionalEmbedding(
            max_num_tiles=max_num_tiles,
            embed_dim=embed_dim,
        )
        # Create state_dict mimicking loading scenario
        state_dict = {
            "model.embedding": params["input_tensor"],
        }

        # Call the hook
        state_dict_copy = state_dict.copy()
        model._load_state_dict_hook(state_dict_copy, "model.")

        # Assert expected outputs
        assert_expected(
            state_dict_copy["model.embedding"],
            params["expected_output"],
            atol=1e-3,
            rtol=1e-4,
        )

        # Check for errors with non-square tile grid sizes
        with pytest.raises(ValueError):
            bad_state_dict = state_dict.copy()
            # Manipulate the tensor to have non-equal max_num_tiles_x and max_num_tiles_y
            bad_tensor = torch.cat(
                (params["input_tensor"], params["input_tensor"][:, [0], :, :]), dim=1
            )
            bad_state_dict["model.embedding"] = bad_tensor
            model._load_state_dict_hook(bad_state_dict, "model.")
