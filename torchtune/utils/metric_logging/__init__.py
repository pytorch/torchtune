# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List

from .disk_logger import DiskLogger
from .metric_logger import MetricLoggerInterface, Scalar
from .stdout_logger import StdoutLogger
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandBLogger

__all__ = [
    "DiskLogger",
    "StdoutLogger",
    "TensorBoardLogger",
    "WandBLogger",
    "MetricLoggerInterface",
    "Scalar",
]


ALL_METRIC_LOGGERS: Dict[str, "MetricLoggerInterface"] = {
    "wandb": WandBLogger,
    "tensorboard": TensorBoardLogger,
    "stdout": StdoutLogger,
    "disk": DiskLogger,
}


def list_metric_loggers() -> List[str]:
    """List available metric loggers.

    Returns:
        List[str]: list of available metric loggers
    """
    return list(ALL_METRIC_LOGGERS.keys())


def get_metric_logger(metric_logger_type: str, **kwargs) -> "MetricLoggerInterface":
    """Get a metric logger based on provided arguments.

    Args:
        metric_logger_type (str): name of the metric logger, options are "wandb", "tensorboard", "stdout", "disk".
        **kwargs: additional arguments to pass to the metric logger

    Raises:
        ValueError: If ``metric_logger`` str is unknown.

    Returns:
        MetricLoggerInterface: metric logger
    """
    if metric_logger_type not in ALL_METRIC_LOGGERS:
        raise ValueError(
            f"Metric logger not recognized. Expected one of {list_metric_loggers}, received {metric_logger_type}."
        )

    return ALL_METRIC_LOGGERS[metric_logger_type](**kwargs)
