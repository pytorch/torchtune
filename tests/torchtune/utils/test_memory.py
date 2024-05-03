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
