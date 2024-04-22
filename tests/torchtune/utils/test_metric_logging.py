# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
from io import StringIO
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_utils import assert_expected, captured_output

from torchtune.utils.metric_logging import (
    CometLogger,
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


def _get_comet_mock():
    mock_experiment = Mock()
    # Manually mock used magic methods to avoid requiring to install comet_ml
    # for tests
    mock_experiment.__internal_api__log_metric__ = Mock()
    mock_experiment.__internal_api__log_metrics__ = Mock()
    mock_experiment.__internal_api__log_parameters__ = Mock()

    return mock_experiment


class TestCometLogger:
    def test_experiment_name(self) -> None:
        mock_experiment = _get_comet_mock()
        with patch("comet_ml.start", return_value=mock_experiment) as mock_start:
            experiment_name = "test_experiment_name"
            logger = CometLogger(
                project="test_project", experiment_name=experiment_name
            )

            assert mock_experiment.set_name.call_count == 1

            mock_experiment.set_name.assert_called_once_with(experiment_name)

    def test_log(self) -> None:
        mock_experiment = _get_comet_mock()
        with patch("comet_ml.start", return_value=mock_experiment) as mock_start:
            logger = CometLogger(project="test_project")
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            assert mock_experiment.__internal_api__log_metric__.call_count == 5
            for i in range(5):
                print(mock_experiment.call_args_list)
                mock_experiment.__internal_api__log_metric__.assert_any_call(
                    "test_log", float(i) ** 2, step=i, framework="torchtune"
                )

    def test_log_dict(self) -> None:
        mock_experiment = _get_comet_mock()
        with patch("comet_ml.start", return_value=mock_experiment) as mock_start:
            logger = CometLogger(project="test_project")
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            mock_experiment.__internal_api__log_metrics__.assert_called_with(
                metric_dict, step=1, framework="torchtune"
            )

    def test_log_config(self) -> None:
        mock_experiment = _get_comet_mock()
        with patch("comet_ml.start", return_value=mock_experiment) as mock_start, patch(
            "omegaconf.OmegaConf.save"
        ) as mock_save:
            logger = CometLogger(project="test_project")
            raw_config = {"a": 1, "b": 2, "checkpointer": {"checkpoint_dir": "."}}
            cfg = OmegaConf.create(raw_config)

            logger.log_config(cfg)

            expected_config_path = Path("torchtune_config.yaml")
            mock_experiment.__internal_api__log_parameters__.assert_called_once_with(
                raw_config, framework="torchtune", source="manual", flatten_nested=True
            )
            mock_experiment.log_asset.assert_called_once_with(
                expected_config_path, file_name="torchtune_config.yaml"
            )
