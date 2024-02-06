#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune.utils.config_utils import (
    get_dataset,
    get_loss,
    get_lr_scheduler,
    get_model,
    get_optimizer,
    get_tokenizer,
)


class TestConfigUtils:
    def test_get_dataset(self):
        _ = get_dataset("AlpacaDataset")
        with pytest.raises(ValueError):
            _ = get_dataset("dummy")

    def test_get_model(self):
        _ = get_model("llama2_7b", device="cpu")
        with pytest.raises(ValueError):
            _ = get_model("dummy")

    def test_get_tokenizer(self):
        _ = get_tokenizer("llama2_tokenizer")
        with pytest.raises(ValueError):
            _ = get_tokenizer("dummy")

    def test_get_lr_scheduler(self):
        optim = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=0.01)
        _ = get_lr_scheduler("cosine_schedule_with_warmup", optim)
        _ = get_lr_scheduler("StepLR", optim)
        with pytest.raises(ValueError):
            _ = get_lr_scheduler("dummy", optim)

    def test_get_optimizer(self):
        model = torch.nn.Linear(1, 1)
        _ = get_optimizer("AdamW", model, lr=0.01)
        with pytest.raises(ValueError):
            _ = get_optimizer("dummy", model, lr=0.01)

    def test_get_loss(self):
        _ = get_loss("CrossEntropyLoss")
        with pytest.raises(ValueError):
            _ = get_loss("dummy")
