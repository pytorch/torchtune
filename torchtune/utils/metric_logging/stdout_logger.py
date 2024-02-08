# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from typing import Mapping

from torchtune.utils.metric_logging.metric_logger import MetricLoggerInterface, Scalar


class StdoutLogger(MetricLoggerInterface):
    """Logger to standard output."""

    def log(self, name: str, data: Scalar, step: int) -> None:
        print(f"Step {step} | {name}:{data}")

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        print(f"Step {step} | ", end="")
        for name, data in payload.items():
            print(f"{name}:{data} ", end="")
        print("\n", end="")

    def __del__(self) -> None:
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.flush()
