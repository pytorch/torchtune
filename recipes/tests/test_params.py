# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from recipes.params import FullFinetuneParams


class TestParams:
    @pytest.fixture
    def params(self):
        return dict(
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
            enable_activation_checkpointing=False,
            enable_fsdp=False,
            cpu_offload=False,
            metric_logger_type="disk",
            resume_from_checkpoint=False,
        )

    def test_bad_model(self, params):
        with pytest.raises(ValueError):
            params["model"] = "dummy"
            _ = FullFinetuneParams(**params)
        with pytest.raises(TypeError):
            params["model"] = ""
            _ = FullFinetuneParams(**params)

    def test_bad_dataset(self, params):
        with pytest.raises(ValueError):
            params["dataset"] = "dummy"
            _ = FullFinetuneParams(**params)

    def test_bad_tokenizer(self, params):
        with pytest.raises(ValueError):
            params["tokenizer"] = "dummy"
            _ = FullFinetuneParams(**params)

    def test_bad_dtype(self, params):
        with pytest.raises(ValueError):
            params["dtype"] = "dummy"
            _ = FullFinetuneParams(**params)

    def test_bad_metric_logger(self, params):
        with pytest.raises(ValueError):
            params["metric_logger_type"] = "dummy"
            _ = FullFinetuneParams(**params)

    def test_cpu_offload_without_cuda(self, params):
        with pytest.raises(ValueError):
            params["cpu_offload"] = True
            params["device"] = "cpu"
            _ = FullFinetuneParams(**params)

    def test_fsdp_not_on_cpu(self, params):
        with pytest.raises(ValueError):
            params["enable_fsdp"] = True
            params["device"] = "cpu"
            _ = FullFinetuneParams(**params)
