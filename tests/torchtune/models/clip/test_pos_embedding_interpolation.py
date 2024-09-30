# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        "tgt_num_tiles": 1,
        # [max_num_tiles, max_num_tiles, 1, embed_dim] -> (2, 2, 2, 3)
        "input_tensor": torch.tensor(
            [
                [
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                    [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                ],
                [
                    [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                    [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                ],
            ]
        ),
        "expected_output": torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]]]),
    },
    {
        "tgt_num_tiles": 3,
        # [max_num_tiles, max_num_tiles, 1, embed_dim] -> (2, 2, 1, 2)
        "input_tensor": torch.tensor(
            [[[[0.0, 1.0]], [[2.0, 3.0]]], [[[4.0, 5.0]], [[6.0, 7.0]]]]
        ),
        "expected_output": torch.tensor(
            [
                [[[0.0, 1.0]], [[1.0, 2.0]], [[2.0, 3.0]]],
                [[[2.0, 3.0]], [[3.0, 4.0]], [[4.0, 5.0]]],
                [[[4.0, 5.0]], [[5.0, 6.0]], [[6.0, 7.0]]],
            ]
        ),
    },
]

local_pos_emb_test_cases = [
    {
        "target_n_tokens_per_tile": 1,
        # [inpt_n_tokens_per_tile, emb_dim] -> (5, 2)
        "input_tensor": torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
        ),
        "expected_output": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
    },
    {
        "target_n_tokens_per_tile": 3,
        # [inpt_n_tokens_per_tile, emb_dim] -> (5, 11)
        "input_tensor": torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        "expected_output": torch.tensor(
            [
                [0.0000],
                [1.0000],
                [1.5000],
                [2.0000],
                [2.0000],
                [2.5000],
                [3.0000],
                [3.0000],
                [3.5000],
                [4.0000],
            ]
        ),
    },
]

global_pos_emb_test_cases = [
    {
        "tgt_max_num_tiles": 2,
        "tgt_patch_grid_size": 2,
        # [max_num_tiles, max_num_tiles, num_tokens_per_tile, embed_dim] -> (3, 3, 2, 1)
        "input_tensor": torch.tensor(
            [
                [[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]],
                [[[6.0], [7.0]], [[8.0], [9.0]], [[10.0], [11.0]]],
                [[[12.0], [13.0]], [[14.0], [15.0]], [[16.0], [17.0]]],
            ]
        ),
        "expected_output": torch.tensor(
            [
                [
                    [[0.0000], [1.0000], [2.3333], [5.0000], [6.3333]],
                    [[4.0000], [3.6667], [5.0000], [7.6667], [9.0000]],
                ],
                [
                    [[12.0000], [9.0000], [10.3333], [13.0000], [14.3333]],
                    [[16.0000], [11.6667], [13.0000], [15.6667], [17.0000]],
                ],
            ]
        ),
    },
    {
        "tgt_max_num_tiles": 1,
        "tgt_patch_grid_size": 1,
        # [max_num_tiles, max_num_tiles, num_tokens_per_tile, embed_dim] -> (1, 1, 5, 2)
        "input_tensor": torch.tensor(
            [[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]]]
        ),
        "expected_output": torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]),
    },
]


class TestPositionalEmbeddingsInterpolation:
    @pytest.mark.parametrize("params", tile_pos_emb_test_cases)
    def test_dynamic_resize(self, params):
        tgt_num_tiles = params["tgt_num_tiles"]
        expected_output = params["expected_output"]
        embedding = params["input_tensor"]

        resized_pos_embed = TilePositionalEmbedding._resize_position_embedding(
            embedding, tgt_num_tiles
        )

        assert_expected(resized_pos_embed, expected_output, atol=1e-3, rtol=1e-4)

    @pytest.mark.parametrize("params", local_pos_emb_test_cases)
    def test_resize_local_position_embedding(self, params):
        input_tensor = params["input_tensor"]
        target_n_tokens_per_tile = params["target_n_tokens_per_tile"]
        expected_output = params["expected_output"]

        resized_pos_embed = (
            TiledTokenPositionalEmbedding._resize_local_position_embedding(
                input_tensor, target_n_tokens_per_tile
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
