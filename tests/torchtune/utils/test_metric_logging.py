# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile
import unittest
from io import StringIO
from typing import cast
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from torchtune.utils.metric_logging import StdoutLogger, TensorBoardLogger, WandBLogger

from tests.test_utils import captured_output


class StdoutLoggerTest(unittest.TestCase):
    def test_stdout_log(self) -> None:
        logger = StdoutLogger()
        with captured_output() as (out, _):
            logger.log(step=0, name="metric_1", data=1.1)
            out = cast(StringIO, out)
            self.assertTrue(
                out.getvalue() == "Step 0 | metric_1:1.1\n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log(step=1, name="metric_1", data=2.1)
            self.assertTrue(
                out.getvalue() == "Step 0 | metric_1:1.1\nStep 1 | metric_1:2.1\n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.close()
            self.assertTrue(
                out.getvalue() == "Step 0 | metric_1:1.1\nStep 1 | metric_1:2.1\n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

    def test_stdout_log_dict(self) -> None:
        logger = StdoutLogger()
        with captured_output() as (out, _):
            logger.log_dict(step=0, payload={"metric_1": 1, "metric_2": 1})
            out = cast(StringIO, out)
            self.assertTrue(
                out.getvalue() == "Step 0 | metric_1:1 metric_2:1 \n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.log_dict(
                step=1, payload={"metric_1": 2, "metric_2": 2.2, "metric_3": 2.2344}
            )
            self.assertTrue(
                out.getvalue()
                == "Step 0 | metric_1:1 metric_2:1 \nStep 1 | metric_1:2 metric_2:2.2 metric_3:2.2344 \n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )

            logger.close()
            self.assertTrue(
                out.getvalue()
                == "Step 0 | metric_1:1 metric_2:1 \nStep 1 | metric_1:2 metric_2:2.2 metric_3:2.2344 \n",
                msg=repr(f"Actual output: {out.getvalue()}"),
            )


class TensorBoardLoggerTest(unittest.TestCase):
    def test_log(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(log_dir=log_dir)
            for i in range(5):
                logger.log("test_log", float(i) ** 2, i)
            logger.close()

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i, event in enumerate(acc.Tensors("test_log")):
                self.assertAlmostEqual(event.tensor_proto.float_val[0], float(i) ** 2)
                self.assertEqual(event.step, i)

    def test_log_dict(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            logger = TensorBoardLogger(log_dir=log_dir)
            metric_dict = {f"log_dict_{i}": float(i) ** 2 for i in range(5)}
            logger.log_dict(metric_dict, 1)
            logger.close()

            acc = EventAccumulator(log_dir)
            acc.Reload()
            for i in range(5):
                tensor_tag = acc.Tensors(f"log_dict_{i}")[0]
                self.assertAlmostEqual(
                    tensor_tag.tensor_proto.float_val[0], float(i) ** 2
                )
                self.assertEqual(tensor_tag.step, 1)


class WandBLoggerTest(unittest.TestCase):
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
