# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
from pathlib import Path

import pytest
import torch
from torch import randn, zeros

from torchtune.training.checkpointing import DistributedCheckpointer
from torchtune.training.seed import set_seed

_VOCAB_SIZE = 100
_DIM = 64
_HIDDEN_DIM = 256


@pytest.fixture(autouse=True)
def random():
    set_seed(16)


class TestDistributedCheckpointer:
    @pytest.fixture
    def weight_dtype(self):
        return torch.float16

    @pytest.fixture
    def state_dict(self, weight_dtype):
        """
        State dict
        """
        state_dict = {
            "model.embed_tokens.weight": randn(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
            "model.layers.0.input_layernorm.weight": randn(_DIM, dtype=weight_dtype),
            "model.layers.0.self_attn.q_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.k_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.v_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.o_proj.weight": randn(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.post_attention_layernorm.weight": randn(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.rotary_emb.inv_freq": randn(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.gate_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.down_proj.weight": randn(
                _DIM, _HIDDEN_DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.up_proj.weight": randn(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.norm.weight": randn(_DIM, dtype=weight_dtype),
            "lm_head.weight": randn(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
        }

        return state_dict

    @pytest.fixture
    def empty_state_dict(self, weight_dtype):
        """
        State dict
        """
        state_dict = {
            "model.embed_tokens.weight": zeros(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
            "model.layers.0.input_layernorm.weight": zeros(_DIM, dtype=weight_dtype),
            "model.layers.0.self_attn.q_proj.weight": zeros(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.k_proj.weight": zeros(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.v_proj.weight": zeros(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.o_proj.weight": zeros(
                _DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.post_attention_layernorm.weight": zeros(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.0.self_attn.rotary_emb.inv_freq": zeros(
                _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.gate_proj.weight": zeros(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.down_proj.weight": zeros(
                _DIM, _HIDDEN_DIM, dtype=weight_dtype
            ),
            "model.layers.0.mlp.up_proj.weight": zeros(
                _HIDDEN_DIM, _DIM, dtype=weight_dtype
            ),
            "model.norm.weight": zeros(_DIM, dtype=weight_dtype),
            "lm_head.weight": zeros(_VOCAB_SIZE, _DIM, dtype=weight_dtype),
        }

        return state_dict

    @pytest.fixture
    def distributed_checkpointer(self, tmp_path) -> DistributedCheckpointer:
        return DistributedCheckpointer(
            checkpoint_dir=tmp_path,
            output_dir=tmp_path,
        )

    def test_save_load_checkpoint(
        self, distributed_checkpointer, state_dict, empty_state_dict
    ):
        """
        Test ``load_checkpoint`` method within the DistributedCheckpointer.

        We test:
        * ``load_checkpoint`` loads the right sets of keys
        * Internal state of the checkpointer is correctly updated.
        """

        distributed_checkpointer.save_checkpoint(
            state_dict=state_dict, epoch=1, save_async=False
        )

        checkpoint_path = Path.joinpath(
            distributed_checkpointer._output_dir,
            f"{distributed_checkpointer._checkpoint_dir_prefix}_1",
        )

        assert os.path.exists(checkpoint_path)

        distributed_checkpointer.load_checkpoint(
            state_dict=empty_state_dict,
        )

        for key in state_dict.keys():
            assert torch.equal(state_dict[key], empty_state_dict[key])

        # clean ups
        shutil.rmtree(checkpoint_path)

    def test_save_load_adapter_checkpoint(
        self, distributed_checkpointer, state_dict, empty_state_dict
    ):
        """
        Test ``load_checkpoint`` method of an adapter checkpoint within the DistributedCheckpointer.
        """
        distributed_checkpointer.save_checkpoint(
            state_dict=state_dict, epoch=1, save_async=False, adapter_only=True
        )
        distributed_checkpointer.save_checkpoint(
            state_dict=state_dict, epoch=1, save_async=False, adapter_only=True
        )

        checkpoint_path = Path.joinpath(
            distributed_checkpointer._output_dir,
            f"{distributed_checkpointer._checkpoint_dir_prefix}_1",
            "adapter_model",
        )

        assert os.path.exists(checkpoint_path)

        distributed_checkpointer.load_checkpoint(
            state_dict=empty_state_dict,
            adapter_only=True,
        )

        for key in state_dict.keys():
            assert torch.equal(state_dict[key], empty_state_dict[key])

        # clean ups
        shutil.rmtree(checkpoint_path)
