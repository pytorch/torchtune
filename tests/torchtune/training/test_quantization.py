# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tests.test_utils import gpu_test
from torch import nn
from torchtune.training.quantization import (
    _SUPPORTS_INT8_MIXED_PRECISION_TRAINING,
    Int8MixedPrecisionTrainingQuantizer,
)


@gpu_test(gpu_count=1)
@pytest.mark.skipif(
    not _SUPPORTS_INT8_MIXED_PRECISION_TRAINING,
    reason="INT8 mixed-precision training is not supported",
)
def test_int8_mixed_precision_training_quantizer():
    quantizer = Int8MixedPrecisionTrainingQuantizer()
    model = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
    ).cuda()
    quantizer.prepare(model)

    # make sure class is changed
    assert model[0].__class__ != nn.Linear
    assert model[2].__class__ != nn.Linear

    # smoke test forward and backward
    model(torch.randn(2, 32).cuda()).sum().backward()
    for p in model.parameters():
        assert p.grad is not None

    # state dict is plain tensor
    state_dict = model.state_dict()
    for v in state_dict.values():
        assert v.__class__ == torch.Tensor
