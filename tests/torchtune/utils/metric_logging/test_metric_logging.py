# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest.mock import patch

from torchtune.utils.metric_logging import (
    DiskLogger,
    get_metric_logger,
    list_metric_loggers,
    StdoutLogger,
    TensorBoardLogger,
    WandBLogger,
)


class TestMetricLogger:
    def test_list_metric_loggers(self) -> None:
        assert set(list_metric_loggers()) == {
            "disk",
            "stdout",
            "tensorboard",
            "wandb",
        }

    def test_get_metric_logger(self) -> None:
        fake_kwargs = {
            "log_dir": "/tmp/output",
            "project": "test-project",
            "extra_key": "bananas",
        }
        assert isinstance(get_metric_logger("disk", **fake_kwargs), DiskLogger)
        assert isinstance(get_metric_logger("stdout", **fake_kwargs), StdoutLogger)
        assert isinstance(
            get_metric_logger("tensorboard", **fake_kwargs), TensorBoardLogger
        )
        with patch("wandb.init") as wandb_init:
            assert isinstance(get_metric_logger("wandb", **fake_kwargs), WandBLogger)
