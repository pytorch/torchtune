# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchtune.modules.model_fusion import register_fusion_module


def test_register_fusion_module():
    """
    Test that all parameters are returned as fusion_params.
    """
    model = nn.Linear(1, 1)
    register_fusion_module(model)

    fusion_params = set(model.fusion_params())
    assert fusion_params == {"weight", "bias"}
