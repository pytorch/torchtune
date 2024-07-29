# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import time
from pathlib import Path

from typing import Mapping, Optional, Union

from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from torchtune.utils import get_logger
from torchtune.utils._distributed import get_world_size_and_rank
from typing_extensions import Protocol

Scalar = Union[Tensor, ndarray, int, float]

log = get_logger("DEBUG")


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

    def log_config(self, config: DictConfig) -> None:
        """Logs the config

        Args:
            config (DictConfig): config to log
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
        filename (Optional[str]): optional filename to write logs to.
            Default: None, in which case log_{unixtimestamp}.txt will be used.
        **kwargs: additional arguments

    Warning:
        This logger is not thread-safe.

    Note:
        This logger creates a new file based on the current time.
    """

    def __init__(self, log_dir: str, filename: Optional[str] = None, **kwargs):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if not filename:
            unix_timestamp = int(time.time())
            filename = f"log_{unix_timestamp}.txt"
        self._file_name = self.log_dir / filename
        self._file = open(self._file_name, "a")
        print(f"Writing logs to {self._file_name}")

    def path_to_log_file(self) -> Path:
        return self._file_name

    def log(self, name: str, data: Scalar, step: int) -> None:
        self._file.write(f"Step {step} | {name}:{data}\n")
        self._file.flush()

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        self._file.write(f"Step {step} | ")
        for name, data in payload.items():
            self._file.write(f"{name}:{data} ")
        self._file.write("\n")
        self._file.flush()

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
        project (str): WandB project name. Default is `torchtune`.
        entity (Optional[str]): WandB entity name. If you don't specify an entity,
            the run will be sent to your default entity, which is usually your username.
        group (Optional[str]): WandB group name for grouping runs together. If you don't
            specify a group, the run will be logged as an individual experiment.
        log_dir (Optional[str]): WandB log directory. If not specified, use the `dir`
            argument provided in kwargs. Else, use root directory.
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
        project: str = "torchtune",
        entity: Optional[str] = None,
        group: Optional[str] = None,
        log_dir: Optional[str] = None,
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

        # Use dir if specified, otherwise use log_dir.
        self.log_dir = kwargs.pop("dir", log_dir)

        _, self.rank = get_world_size_and_rank()

        if self._wandb.run is None and self.rank == 0:
            # we check if wandb.init got called externally,
            run = self._wandb.init(
                project=project,
                entity=entity,
                group=group,
                dir=self.log_dir,
                **kwargs,
            )

        if self._wandb.run:
            self._wandb.run._label(repo="torchtune")

        # define default x-axis (for latest wandb versions)
        if getattr(self._wandb, "define_metric", None):
            self._wandb.define_metric("global_step")
            self._wandb.define_metric("*", step_metric="global_step", step_sync=True)

        self.config_allow_val_change = kwargs.get("allow_val_change", False)

    def log_config(self, config: DictConfig) -> None:
        """Saves the config locally and also logs the config to W&B. The config is
        stored in the same directory as the checkpoint. You can
        see an example of the logged config to W&B in the following link:
        https://wandb.ai/capecape/torchtune/runs/6053ofw0/files/torchtune_config_j67sb73v.yaml

        Args:
            config (DictConfig): config to log
        """
        if self._wandb.run:
            resolved = OmegaConf.to_container(config, resolve=True)
            self._wandb.config.update(
                resolved, allow_val_change=self.config_allow_val_change
            )
            try:
                output_config_fname = Path(
                    os.path.join(
                        config.checkpointer.checkpoint_dir,
                        "torchtune_config.yaml",
                    )
                )
                OmegaConf.save(config, output_config_fname)

                log.info(f"Logging {output_config_fname} to W&B under Files")
                self._wandb.save(
                    output_config_fname, base_path=output_config_fname.parent
                )

            except Exception as e:
                log.warning(
                    f"Error saving {output_config_fname} to W&B.\nError: \n{e}."
                    "Don't worry the config will be logged the W&B workspace"
                )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._wandb.run:
            self._wandb.log({name: data, "global_step": step})

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self._wandb.run:
            self._wandb.log({**payload, "global_step": step})

    def __del__(self) -> None:
        if self._wandb.run:
            self._wandb.finish()

    def close(self) -> None:
        if self._wandb.run:
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
