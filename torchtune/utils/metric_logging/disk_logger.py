# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path

from typing import Mapping

from torchtune.utils.metric_logging.metric_logger import MetricLoggerInterface, Scalar


class DiskLogger(MetricLoggerInterface):
    """Logger to disk.

    Args:
        log_dir (str): directory to store logs
        **kwargs: additional arguments

    Warning:
        This logger is not thread-safe.

    Note:
        This logger creates a new file based on the current time.
    """

    def __init__(self, log_dir: str, **kwargs):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        unix_timestamp = int(time.time())
        self._file_name = self.log_dir / f"log_{unix_timestamp}.txt"
        self._file = open(self._file_name, "a")
        print(f"Writing logs to {self._file_name}")

    def path_to_log_file(self) -> Path:
        return self._file_name

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._file.write(f"Step {step} | {name}:{data}\n")

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._file.write(f"Step {step} | ")
        for name, data in payload.items():
            self._file.write(f"{name}:{data} ")
        self._file.write("\n")

    def __del__(self) -> None:
        self._file.close()

    def close(self) -> None:
        self._file.close()
