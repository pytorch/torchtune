# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Mapping, Optional

from torchtune.utils.metric_logging.metric_logger import MetricLoggerInterface, Scalar


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
