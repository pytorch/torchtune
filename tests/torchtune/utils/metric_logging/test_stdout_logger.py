# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from io import StringIO
from typing import cast

from tests.test_utils import captured_output

from torchtune.utils.metric_logging import StdoutLogger


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
