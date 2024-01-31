# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from pathlib import Path

from typing import Dict, List, Mapping, Optional, Union

from numpy import ndarray
from torch import Tensor
from typing_extensions import Protocol

from torchtune.utils.distributed import get_world_size_and_rank

Scalar = Union[Tensor, ndarray, int, float]


class MetricLoggerInterface(Protocol):
    """Abstract metric logger."""

    def log(
        self,
        name: str,
        data: Scalar,
        step: int,
    ) -> None:
        """Log scalar data.

        Args:
            name (str): tag name used to group scalars
            data (Scalar): scalar data to log
            step (int): step value to record
        """
        pass

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        """Log multiple scalar values.

        Args:
            payload (Mapping[str, Scalar]): dictionary of tag name and scalar value
            step (int): step value to record
        """
        pass

    def close(self) -> None:
        """
        Close log resource, flushing if necessary.
        Logs should not be written after `close` is called.
        """
        pass


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


class WandBLogger(MetricLoggerInterface):
    """Logger for use w/ Weights and Biases application (https://wandb.ai/).
    For more information about arguments expected by WandB, see https://docs.wandb.ai/ref/python/init.

    Args:
        project (str): WandB project name
        entity (Optional[str]): WandB entity name
        group (Optional[str]): WandB group name
        **kwargs: additional arguments to pass to wandb.init

    Example:
        >>> from torchtune.utils.metric_logging import WandBLogger
        >>> logger = WandBLogger(project="my_project", entity="my_entity", group="my_group")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Raises:
        ImportError: If ``wandb`` package is not installed.

    Note:
        This logger requires the wandb package to be installed.
        You can install it with `pip install wandb`.
        In order to use the logger, you need to login to your WandB account.
        You can do this by running `wandb login` in your terminal.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        **kwargs,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "``wandb`` package not found. Please install wandb using `pip install wandb` to use WandBLogger."
                "Alternatively, use the ``StdoutLogger``, which can be specified by setting metric_logger_type='stdout'."
            ) from e

        self._wandb = wandb
        self._wandb.init(
            project=project,
            entity=entity,
            group=group,
            reinit=True,
            resume="allow",
            config=kwargs,
        )

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._wandb.log({name: data}, step=step)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._wandb.log(payload, step=step)

    def __del__(self) -> None:
        self._wandb.finish()

    def close(self) -> None:
        self._wandb.finish()


class TensorBoardLogger(MetricLoggerInterface):
    """Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        log_dir (str): TensorBoard log directory
        organize_logs (bool): If `True`, this class will create a subdirectory within `log_dir` for the current
            run. Having sub-directories allows you to compare logs across runs. When TensorBoard is
            passed a logdir at startup, it recursively walks the directory tree rooted at logdir looking for
            subdirectories that contain tfevents data. Every time it encounters such a subdirectory,
            it loads it as a new run, and the frontend will organize the data accordingly.
            Recommended value is `True`. Run `tensorboard --logdir my_log_dir` to view the logs.
        **kwargs: additional arguments

    Example:
        >>> from torchtune.utils.metric_logging import TensorBoardLogger
        >>> logger = TensorBoardLogger(log_dir="my_log_dir")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Note:
        This utility requires the tensorboard package to be installed.
        You can install it with `pip install tensorboard`.
        In order to view TensorBoard logs, you need to run `tensorboard --logdir my_log_dir` in your terminal.
    """

    def __init__(self, log_dir: str, organize_logs: bool = True, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_world_size_and_rank()

        # In case organize_logs is `True`, update log_dir to include a subdirectory for the
        # current run
        self.log_dir = (
            os.path.join(log_dir, f"run_{self._rank}_{time.time()}")
            if organize_logs
            else log_dir
        )

        # Initialize the log writer only if we're on rank 0.
        if self._rank == 0:
            self._writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self.log(name, data, step)

    def __del__(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            self._writer = None


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
