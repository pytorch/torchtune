# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from recipes.params import FullFinetuneParams


class TestParams:
    def test_bad_post_init(self):
        params = dict(
            dataset="alpaca",
            seed=None,
            shuffle=True,
            model="llama2_7b",
            model_checkpoint="/tmp/llama2-7b",
            tokenizer="llama2_tokenizer",
            tokenizer_checkpoint="/tmp/tokenizer.model",
            batch_size=2,
            lr=2e-5,
            epochs=3,
            optimizer="SGD",
            loss="CrossEntropyLoss",
            output_dir="/tmp/alpaca-llama2-finetune",
            device="cuda",
            dtype="fp32",
            activation_checkpointing=False,
            cpu_offload=False,
            metric_logger_type="disk",
            resume_from_previous_checkpoint=False,
        )
        with pytest.raises(ValueError):
            params["model"] = "dummy"
            _ = FullFinetuneParams(**params)
        with pytest.raises(ValueError):
            params["dataset"] = "dummy"
            _ = FullFinetuneParams(**params)
        with pytest.raises(ValueError):
            params["tokenizer"] = "dummy"
            _ = FullFinetuneParams(**params)
        with pytest.raises(ValueError):
            params["dtype"] = "dummy"
            _ = FullFinetuneParams(**params)
        with pytest.raises(ValueError):
            params["metric_logger_type"] = "dummy"
            _ = FullFinetuneParams(**params)
