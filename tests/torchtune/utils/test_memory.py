# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torchtune import modules, utils
from torchtune.models.llama2._component_builders import llama2
from torchtune.models.llama3._component_builders import llama3
from torchtune.utils import set_activation_checkpointing


class TestSetActivationCheckpointing:
    @pytest.fixture
    def model(self) -> int:
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def _verify(self, model):
        for submodule in model.modules():
            if isinstance(submodule, CheckpointWrapper):
                assert isinstance(submodule._checkpoint_wrapped_module, nn.Linear)

    def test_activation_checkpoint_set_policy(self, model):
        set_activation_checkpointing(model=model, auto_wrap_policy={nn.Linear})
        self._verify(model)

    def test_activation_checkpoint_custom_policy(self, model):
        def custom_policy(module: nn.Module, recurse: bool, **kwargs) -> bool:
            if recurse:
                return True
            return isinstance(module, nn.Linear)

        set_activation_checkpointing(model=model, auto_wrap_policy=custom_policy)
        self._verify(model)

    def test_get_ac_policy(self):
        l3 = llama3(
            vocab_size=64,
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            embed_dim=64,
            max_seq_len=128,
        )
        l2 = llama2(
            vocab_size=64,
            num_layers=1,
            num_heads=4,
            num_kv_heads=4,
            embed_dim=64,
            max_seq_len=128,
        )

        set_activation_checkpointing(
            l3,
            auto_wrap_policy=utils.get_ac_policy(
                "LLAMA3", {modules.TransformerDecoderLayer}
            ),
        )
        set_activation_checkpointing(
            l2,
            auto_wrap_policy=utils.get_ac_policy(
                "LLAMA2", {modules.TransformerDecoderLayer}
            ),
        )
        assert isinstance(l3.tok_embeddings, CheckpointWrapper)
        assert not isinstance(l2.tok_embeddings, CheckpointWrapper)
        assert isinstance(l3.output, CheckpointWrapper)
        assert not isinstance(l2.output, CheckpointWrapper)
        for layer in l3.layers:
            assert isinstance(layer, CheckpointWrapper)

        for layer in l2.layers:
            assert isinstance(layer, CheckpointWrapper)
