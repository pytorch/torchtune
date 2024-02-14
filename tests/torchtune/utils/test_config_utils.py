#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtune import datasets, models, tokenizers
from torchtune.modules import lr_schedulers
from torchtune.utils import metric_logging
from torchtune.utils.config_utils import (
    get_dataset,
    get_loss,
    get_lr_scheduler,
    get_metric_logger,
    get_model,
    get_optimizer,
    get_tokenizer,
)


class TestConfigUtils:
    def test_get_dataset(self):
        datasets.foo = lambda x: x
        dataset = get_dataset("foo", x=1)
        assert dataset == 1
        with pytest.raises(ValueError):
            _ = get_dataset("dummy")

    def test_get_model(self):
        models.foo = lambda x: x
        model = get_model("foo", device=torch.device("cpu"), x=1)
        assert model == 1

        with pytest.raises(ValueError):
            _ = get_model("dummy", device=torch.device("cpu"))

    def test_get_tokenizer(self):
        tokenizers.foo = lambda x: x
        tokenizer = get_tokenizer("foo", x=1)
        assert tokenizer == 1

        with pytest.raises(ValueError):
            _ = get_tokenizer("dummy")

    def test_get_lr_scheduler(self):
        optim = torch.optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=0.01)
        lr_schedulers.foo = lambda x, **kwargs: x
        lr_scheduler = get_lr_scheduler("foo", optimizer=optim, x=1)
        assert lr_scheduler == 1

        _ = get_lr_scheduler("StepLR", optimizer=optim, step_size=10)

        with pytest.raises(ValueError):
            _ = get_lr_scheduler("dummy", optimizer=optim)

    def test_get_optimizer(self):
        model = torch.nn.Linear(1, 1)
        _ = get_optimizer("AdamW", model, lr=0.01)
        with pytest.raises(ValueError):
            _ = get_optimizer("dummy", model, lr=0.01)

    def test_get_loss(self):
        _ = get_loss("CrossEntropyLoss")
        with pytest.raises(ValueError):
            _ = get_loss("dummy")

    def test_get_metric_logger(self):
        metric_logging.foo = lambda x: x
        logger = get_metric_logger("foo", x=1)
        assert logger == 1

        with pytest.raises(ValueError):
            _ = get_metric_logger("dummy")
