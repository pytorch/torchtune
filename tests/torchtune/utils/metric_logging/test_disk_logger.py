# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile

from tests.test_utils import assert_expected

from torchtune.utils.metric_logging import DiskLogger


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
