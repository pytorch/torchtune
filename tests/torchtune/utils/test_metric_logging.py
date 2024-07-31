# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
from io import StringIO
from typing import cast
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_utils import assert_expected, captured_output

from torchtune.utils.metric_logging import (
    ClearMLLogger,
    DiskLogger,
    StdoutLogger,
    TensorBoardLogger,
    WandBLogger,
)


class TestDiskLogger:
    def test_log(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DiskLogger(log_dir=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            log_path = logger.path_to_log_file()
            assert log_path.exists()
            values = open(log_path).readlines()
            assert_expected(len(values), 5)
            for i in range(5):
                assert values[i] == f"Step {i} | test_log:{float(i) ** 2}\n"

    def test_log_dict(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = DiskLogger(log_dir=log_dir)
            for i in range(5):
                logger.log_dict(step=i, payload={"metric_1": i, "metric_2": i**2})
            logger.close()

            log_path = logger.path_to_log_file()
            assert log_path.exists()
            values = open(log_path).readlines()
            assert_expected(len(values), 5)
            for i in range(5):
                assert values[i] == f"Step {i} | metric_1:{i} metric_2:{i ** 2} \n"


class TestStdoutLogger:
    def test_stdout_log(self) -> None:
        logger = StdoutLogger()
        with captured_output() as (out, _):
            logger.log(step=0, name="metric_1", data=1.1)
            out = cast(StringIO, out)
            assert (
                out.getvalue() == "Step 0 | metric_1:1.1\n"
            ), f"Actual output: {out.getvalue()}"

            logger.log(step=1, name="metric_1", data=2.1)
            assert (
                out.getvalue() == "Step 0 | metric_1:1.1\nStep 1 | metric_1:2.1\n"
            ), f"Actual output: {out.getvalue()}"

            logger.close()
            assert (
                out.getvalue() == "Step 0 | metric_1:1.1\nStep 1 | metric_1:2.1\n"
            ), f"Actual output: {out.getvalue()}"

    def test_stdout_log_dict(self) -> None:
        logger = StdoutLogger()
        with captured_output() as (out, _):
            logger.log_dict(step=0, payload={"metric_1": 1, "metric_2": 1})
            out = cast(StringIO, out)
            assert (
                out.getvalue() == "Step 0 | metric_1:1 metric_2:1 \n"
            ), f"Actual output: {out.getvalue()}"

            logger.log_dict(
                step=1, payload={"metric_1": 2, "metric_2": 2.2, "metric_3": 2.2344}
            )
            assert (
                out.getvalue()
                == "Step 0 | metric_1:1 metric_2:1 \nStep 1 | metric_1:2 metric_2:2.2 metric_3:2.2344 \n"
            ), f"Actual output: {out.getvalue()}"

            logger.close()
            assert (
                out.getvalue()
                == "Step 0 | metric_1:1 metric_2:1 \nStep 1 | metric_1:2 metric_2:2.2 metric_3:2.2344 \n"
            ), f"Actual output: {out.getvalue()}"


class TestTensorBoardLogger:
    def test_log(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(log_dir=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            acc = EventAccumulator(logger.log_dir)
            acc.Reload()
            for i, event in enumerate(acc.Tensors("test_log")):
                assert_expected(event.tensor_proto.float_val[0], float(i) ** 2)
                assert_expected(event.step, i)

    def test_log_dict(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(log_dir=log_dir)
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            acc = EventAccumulator(logger.log_dir)
            acc.Reload()
            for i in range(5):
                tensor_tag = acc.Tensors(f"log_dict_{i}")[0]
                assert_expected(tensor_tag.tensor_proto.float_val[0], float(i) ** 2)
                assert_expected(tensor_tag.step, 1)


@pytest.mark.skip(reason="This was never running and needs to be fixed")
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

    def test_save_config(self) -> None:
        with patch("wandb.init") as mock_init, patch(
            "wandb.run", create=True
        ) as mock_run, patch("OmegaConf.save") as mock_save, patch(
            "wandb.save"
        ) as mock_wandb_save:

            logger = WandBLogger(project="test_project")
            cfg = OmegaConf.create({"a": 1, "b": 2})

            with patch.object(logger, "_wandb", mock_run):
                logger.save_config(cfg)

            expected_config_path = "torchtune_config.yaml"
            mock_save.assert_called_once_with(cfg, expected_config_path)
            mock_wandb_save.assert_called_once_with(expected_config_path)


@pytest.mark.skip(reason="Will fail if clearml is not installed")
class TestClearmlLogger:
    def test_clearml_import(self):
        # Test to ensure that the ClearmlLogger handles the absence of `clearml` gracefully
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'clearml'")
        ):
            with pytest.raises(ImportError):
                logger = ClearMLLogger(project="test_project")

    @patch("clearml.Task.create")
    def test_log_dict(self, mock_create):
        from unittest.mock import MagicMock

        # Setting up the ClearML task and logger mocks
        mock_task_instance = MagicMock()
        mock_logger = MagicMock()
        mock_create.return_value = mock_task_instance
        mock_task_instance.get_logger.return_value = mock_logger

        # Instantiate the logger
        logger = ClearMLLogger(project="test_project")
        metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
        step = 1
        series = None  # Assuming you are using the default series for all entries

        # Call the method under test
        logger.log_dict(metric_dict, step, series)

        # Verify that report_scalar was called correctly for each entry in the dictionary
        for name, value in metric_dict.items():
            mock_logger.report_scalar.assert_any_call(
                title=name, series=series, value=value, iteration=step
            )

        # Ensure the correct number of calls were made
        assert mock_logger.report_scalar.call_count == len(metric_dict)

        # Cleanup by closing the logger
        logger.close()
