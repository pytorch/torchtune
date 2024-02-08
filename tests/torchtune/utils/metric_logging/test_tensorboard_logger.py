# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import tempfile

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_utils import assert_expected

from torchtune.utils.metric_logging import TensorBoardLogger


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
