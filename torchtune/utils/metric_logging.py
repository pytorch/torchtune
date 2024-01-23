# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys

from typing import Mapping, Optional, Union

from numpy import ndarray
from torch import Tensor
from typing_extensions import Protocol

Scalar = Union[Tensor, ndarray, int, float]


def get_metric_logger(
    metric_logger: str, project: Optional[str] = None, log_dir: Optional[str] = None
) -> "MetricLogger":
    """Get a metric logger based on provided arguments.

    Args:
        metric_logger (str): name of the metric logger, options are "wandb", "tensorboard", "stdout".
        project (Optional[str]): WandB project name
        log_dir (Optional[str]): TensorBoard log directory

    Returns:
        MetricLogger: metric logger
    """
    if metric_logger == "wandb":
        return WandBLogger(project=project)
    elif metric_logger == "tensorboard":
        return TensorBoardLogger(log_dir=log_dir)
    elif metric_logger == "stdout":
        return StdoutLogger()
    else:
        raise ValueError(f"Metric logger not recognized. Expected 'wandb', 'tensorboard', or 'stdout', received {metric_logger}.")


class MetricLogger(Protocol):
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


class StdoutLogger(MetricLogger):
    """Metric logger to standard output. This is the default logger."""

    def log(self, name: str, data: Scalar, step: int) -> None:
        print(f"Step {step} | {name}:{data}")

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        print(f"Step {step} | ", end="")
        for name, data in payload.items():
            print(f"{name}:{data} ", end="")
        print("\n", end="")

    def close(self) -> None:
        sys.stdout.flush()


class WandBLogger(MetricLogger):
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
        import wandb

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

    def close(self) -> None:
        self._wandb.finish()


class TensorBoardLogger(MetricLogger):
    """Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        log_dir (str): TensorBoard log directory
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

    def __init__(self, log_dir: str, **kwargs):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir, **kwargs)

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self.log(name, data, step)

    def close(self) -> None:
        self._writer.close()
