# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest.mock import patch

from torchtune.utils.metric_logging import WandBLogger


class TestWandBLogger:
    def test_log(self) -> None:
        with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log:
            logger = WandBLogger(project="test_project")
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            assert mock_log.call_count == 5
            for i in range(5):
                mock_log.assert_any_call({"test_log": float(i) ** 2}, step=i)

    def test_log_dict(self) -> None:
        with patch("wandb.init") as mock_init, patch("wandb.log") as mock_log:
            logger = WandBLogger(project="test_project")
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            mock_log.assert_called_with(metric_dict, step=1)
