# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import sys
import time
from pathlib import Path

from typing import Any, Mapping, Optional, Union

import torch

from numpy import ndarray
from omegaconf import DictConfig, OmegaConf

from torchtune.utils import get_logger, get_world_size_and_rank
from typing_extensions import Protocol

Scalar = Union[torch.Tensor, ndarray, int, float]

log = get_logger("DEBUG")


def save_config(config: DictConfig) -> Path:
    """
    Save the OmegaConf configuration to a YAML file at `{config.output_dir}/torchtune_config.yaml`.

    Args:
        config (DictConfig): The OmegaConf config object to be saved. It must contain an `output_dir` attribute
            specifying where the configuration file should be saved.

    Returns:
        Path: The path to the saved configuration file.

    Note:
        If the specified `output_dir` does not exist, it will be created.
    """
    try:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_config_fname = output_dir / "torchtune_config.yaml"
        OmegaConf.save(config, output_config_fname)
        return output_config_fname
    except Exception as e:
        log.warning(f"Error saving config.\nError: \n{e}.")


def flatten_dict(d: dict[str, Any], *, sep: str = ".", parent_key: str = ""):
    """Recursively flattens a nested dictionary into one level of key-value pairs.

    Args:
        d (dict[str, Any]): Any dictionary to flatten.
        sep (str, optional): Desired separator for flattening nested keys. Defaults to ".".
        parent_key (str, optional): Key prefix for children (nested keys), containing parent key names. Defaults to "".

    Example:
        >>> flatten_dict({"foo": {"bar": "baz"}, "qux": "quux"}, sep="--")
        {"foo--bar": "baz", "qux": "quux"}

    Returns:
        dict[str, Any]: Flattened dictionary.

    Note:
        Does not unnest dictionaries within list values (i.e., {"foo": [{"bar": "baz"}]}).
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, sep=sep, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


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
        """Logs the config as file

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
        output_fmt (str): format of the output file. Default: 'txt'.
            Supported formats: 'txt', 'jsonl'.
        filename (Optional[str]): optional filename to write logs to.
            Default: None, in which case log_{unixtimestamp}.txt will be used.
        **kwargs: additional arguments

    Warning:
        This logger is not thread-safe.

    Note:
        This logger creates a new file based on the current time.
    """

    def __init__(
        self,
        log_dir: str,
        output_fmt: str = "txt",
        filename: Optional[str] = None,
        **kwargs,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_fmt = output_fmt
        assert self.output_fmt in [
            "txt",
            "jsonl",
        ], f"Unsupported output format: {self.output_fmt}. Supported formats: 'txt', 'jsonl'."
        if not filename:
            unix_timestamp = int(time.time())
            filename = f"log_{unix_timestamp}.{self.output_fmt}"
        self._file_name = self.log_dir / filename
        self._file = open(self._file_name, "a")
        print(f"Writing logs to {self._file_name}")

    def path_to_log_file(self) -> Path:
        return self._file_name

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.output_fmt == "txt":
            self._file.write(f"Step {step} | {name}:{data}\n")
        elif self.output_fmt == "jsonl":
            json.dump(
                {"step": step, name: data},
                self._file,
                default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else str(x),
            )
            self._file.write("\n")
        else:
            raise ValueError(
                f"Unsupported output format: {self.output_fmt}. Supported formats: 'txt', 'jsonl'."
            )
        self._file.flush()

    def log_config(self, config: DictConfig) -> None:
        _ = save_config(config)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self.output_fmt == "txt":
            self._file.write(f"Step {step} | ")
            for name, data in payload.items():
                self._file.write(f"{name}:{data} ")
        elif self.output_fmt == "jsonl":
            json.dump(
                {"step": step} | {name: data for name, data in payload.items()},
                self._file,
                default=lambda x: x.tolist() if isinstance(x, torch.Tensor) else str(x),
            )
        else:
            raise ValueError(
                f"Unsupported output format: {self.output_fmt}. Supported formats: 'txt', 'jsonl'."
            )
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

    def log_config(self, config: DictConfig) -> None:
        _ = save_config(config)

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
        >>> from torchtune.training.metric_logging import WandBLogger
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

        # create log_dir if missing
        if self.log_dir is not None and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

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

            # Also try to save the config as a file
            output_config_fname = save_config(config)
            try:
                self._wandb.save(
                    output_config_fname, base_path=output_config_fname.parent
                )
            except Exception as e:
                log.warning(
                    f"Error uploading {output_config_fname} to W&B.\nError: \n{e}."
                    "Don't worry the config will be logged the W&B workspace"
                )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._wandb.run:
            self._wandb.log({name: data, "global_step": step})

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self._wandb.run:
            self._wandb.log({**payload, "global_step": step})

    def __del__(self) -> None:
        # extra check for when there is an import error
        if hasattr(self, "_wandb") and self._wandb.run:
            self._wandb.finish()

    def close(self) -> None:
        if self._wandb.run:
            self._wandb.finish()


class TensorBoardLogger(MetricLoggerInterface):
    """Logger for use w/ PyTorch's implementation of TensorBoard (https://pytorch.org/docs/stable/tensorboard.html).

    Args:
        log_dir (str): torch.TensorBoard log directory
        organize_logs (bool): If `True`, this class will create a subdirectory within `log_dir` for the current
            run. Having sub-directories allows you to compare logs across runs. When TensorBoard is
            passed a logdir at startup, it recursively walks the directory tree rooted at logdir looking for
            subdirectories that contain tfevents data. Every time it encounters such a subdirectory,
            it loads it as a new run, and the frontend will organize the data accordingly.
            Recommended value is `True`. Run `tensorboard --logdir my_log_dir` to view the logs.
        **kwargs: additional arguments

    Example:
        >>> from torchtune.training.metric_logging import TensorBoardLogger
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

    def log_config(self, config: DictConfig) -> None:
        _ = save_config(config)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        for name, data in payload.items():
            self.log(name, data, step)

    def __del__(self) -> None:
        # extra check for when there is an import error
        if hasattr(self, "_writer"):
            self._writer.close()
            self._writer = None

    def close(self) -> None:
        if hasattr(self, "_writer"):
            self._writer.close()
            self._writer = None


class CometLogger(MetricLoggerInterface):
    """Logger for use w/ Comet (https://www.comet.com/site/).
    Comet is an experiment tracking tool that helps ML teams track, debug,
    compare, and reproduce their model training runs.

    For more information about arguments expected by Comet, see
    https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#for-the-experiment.

    Args:
        api_key (Optional[str]): Comet API key. It's recommended to configure the API Key with `comet login`.
        workspace (Optional[str]): Comet workspace name. If not provided, uses the default workspace.
        project (Optional[str]): Comet project name. Defaults to Uncategorized.
        experiment_key (Optional[str]): The key for comet experiment to be used for logging. This is used either to
            append data to an Existing Experiment or to control the ID of new experiments (for example to match another
            ID). Must be an alphanumeric string whose length is between 32 and 50 characters.
        mode (Optional[str]): Control how the Comet experiment is started.

            * ``"get_or_create"``: Starts a fresh experiment if required, or persists logging to an existing one.
            * ``"get"``: Continue logging to an existing experiment identified by the ``experiment_key`` value.
            * ``"create"``: Always creates of a new experiment, useful for HPO sweeps.
        online (Optional[bool]): If True, the data will be logged to Comet server, otherwise it will be stored locally
            in an offline experiment. Default is ``True``.
        experiment_name (Optional[str]): Name of the experiment. If not provided, Comet will auto-generate a name.
        tags (Optional[list[str]]): Tags to associate with the experiment.
        log_code (bool): Whether to log the source code. Defaults to True.
        **kwargs (dict[str, Any]): additional arguments to pass to ``comet_ml.start``. See
            https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment-Creation/#comet_ml.ExperimentConfig

    Example:
        >>> from torchtune.training.metric_logging import CometLogger
        >>> logger = CometLogger(project_name="my_project", workspace="my_workspace")
        >>> logger.log("my_metric", 1.0, 1)
        >>> logger.log_dict({"my_metric": 1.0}, 1)
        >>> logger.close()

    Raises:
        ImportError: If ``comet_ml`` package is not installed.

    Note:
        This logger requires the comet_ml package to be installed.
        You can install it with ``pip install comet_ml``.
        You need to set up your Comet.ml API key before using this logger.
        You can do this by calling ``comet login`` in your terminal.
        You can also set it as the `COMET_API_KEY` environment variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        experiment_key: Optional[str] = None,
        mode: Optional[str] = None,
        online: Optional[bool] = None,
        experiment_name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        log_code: bool = True,
        **kwargs: dict[str, Any],
    ):
        try:
            import comet_ml
        except ImportError as e:
            raise ImportError(
                "``comet_ml`` package not found. Please install comet_ml using `pip install comet_ml` to use CometLogger."
                "Alternatively, use the ``StdoutLogger``, which can be specified by setting metric_logger_type='stdout'."
            ) from e

        # Remove 'log_dir' from kwargs as it is not a valid argument for comet_ml.ExperimentConfig
        if "log_dir" in kwargs:
            del kwargs["log_dir"]

        _, self.rank = get_world_size_and_rank()

        # Declare it early so further methods don't crash in case of
        # Experiment Creation failure due to mis-named configuration for
        # example
        self.experiment = None
        if self.rank == 0:
            self.experiment = comet_ml.start(
                api_key=api_key,
                workspace=workspace,
                project=project,
                experiment_key=experiment_key,
                mode=mode,
                online=online,
                experiment_config=comet_ml.ExperimentConfig(
                    log_code=log_code, tags=tags, name=experiment_name, **kwargs
                ),
            )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.experiment is not None:
            self.experiment.log_metric(name, data, step=step)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self.experiment is not None:
            self.experiment.log_metrics(payload, step=step)

    def log_config(self, config: DictConfig) -> None:
        if self.experiment is not None:
            resolved = OmegaConf.to_container(config, resolve=True)
            self.experiment.log_parameters(resolved)

            # Also try to save the config as a file
            output_config_fname = save_config(config)
            try:
                self.experiment.log_asset(
                    output_config_fname, file_name=output_config_fname.name
                )
            except Exception as e:
                log.warning(f"Failed to upload config to Comet assets. Error: {e}")

    def close(self) -> None:
        if self.experiment is not None:
            self.experiment.end()

    def __del__(self) -> None:
        self.close()


class MLFlowLogger(MetricLoggerInterface):
    """Logger for use w/ MLFlow (https://mlflow.org/).

    Args:
        experiment_name (Optional[str]): MLFlow experiment name. If not specified, will
            default to MLFLOW_EXPERIMENT_NAME environment variable if set, or default.
        tracking_uri (Optional[str]): MLFlow tracking uri. If not specified, will default
            to MLFLOW_TRACKING_URI environment variable if set, or default.
        run_id (Optional[str]): MLFlow run name. If not specified, will default
            to mlflow-generated HRID. Unused if run_id is specified or MLFLOW_RUN_ID
            environment variable is found.
        run_name (Optional[str]): MLFlow run ID. If not specified, will default
            to MLFLOW_RUN_ID environment variable if set, or a new run will be created.

    Example:
        >>> logger = MLFlowLogger(experiment_name="my_experiment", run_name="run1")
        >>> logger.log("accuracy", 0.95, step=1)
        >>> logger.log_dict({"loss": 0.1, "accuracy": 0.95}, step=1)
        >>> logger.log_config(config)
        >>> logger.close()

    Raises:
        ImportError: If ``mlflow`` package is not installed.

    Note:
        This logger requires the mlflow package to be installed.
        You can install it with `pip install mlflow`.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        run_id: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        try:
            import mlflow
        except ImportError as e:
            raise ImportError(
                "``mlflow`` package not found. Please install mlflow using `pip install mlflow` to use MLFlowLogger."
                "Alternatively, use the ``StdoutLogger``, which can be specified by setting metric_logger_type='stdout'."
            ) from e

        _, self.rank = get_world_size_and_rank()

        self._mlflow = mlflow

        self._tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self._experiment_name = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME")
        self._run_id = run_id or os.getenv("MLFLOW_RUN_ID")

        if self.rank == 0:
            if not self._mlflow.is_tracking_uri_set():
                if self._tracking_uri is not None:
                    self._mlflow.set_tracking_uri(self._tracking_uri)

            if self._mlflow.active_run() is None or self._nested_run or self._run_id:
                if self._experiment_name is not None:
                    # Use of set_experiment() ensure that Experiment is created if not exists
                    self._mlflow.set_experiment(self._experiment_name)
                run = self._mlflow.start_run(run_name=run_name)
                self._run_id = run.info.run_id

    def log_config(self, config: DictConfig) -> None:
        """Saves the config locally and also logs the config to mlflow. The config is
        stored in the same directory as the checkpoint.

        Args:
            config (DictConfig): config to log
        """
        if self._mlflow.active_run():
            resolved = OmegaConf.to_container(config, resolve=True)

            # mlflow's params must be flat key-value pairs
            config_as_params = flatten_dict(resolved)
            self._mlflow.log_params(config_as_params, run_id=self._run_id)

            output_config_fname = save_config(config)

            # this avoids break if config's output_dir is an absolute path
            artifact_path = str(output_config_fname.parent).lstrip("/")

            self._mlflow.log_artifact(
                output_config_fname,
                artifact_path=artifact_path,
                run_id=self._run_id,
            )

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._mlflow.active_run():
            self._mlflow.log_metric(name, data, step=step, run_id=self._run_id)

    def log_dict(self, payload: Mapping[str, Scalar], step: int) -> None:
        if self._mlflow.active_run():
            self._mlflow.log_metrics(payload, step=step, run_id=self._run_id)

    def close(self) -> None:
        """
        Ends the MLflow run.
        After calling close, no further logging should be performed.
        """
        if self.rank == 0 and self._mlflow.active_run():
            self._mlflow.end_run()

    def __del__(self) -> None:
        # Ensure the MLflow run is closed when the logger is deleted.
        if hasattr(self, "_mlflow") and self._mlflow.active_run():
            self._mlflow.end_run()
