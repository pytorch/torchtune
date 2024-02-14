# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import ModuleType
from typing import Any, Callable, Union

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset

from torchtune import datasets, models, tokenizers
from torchtune.modules import lr_schedulers
from torchtune.utils import metric_logging
from torchtune.utils.device import get_device
from torchtune.utils.metric_logging import MetricLoggerInterface


def _raise_not_valid_object(name: str, module_path: str) -> None:
    raise ValueError(f"{name} is not a valid object from {module_path}")


def _get_torchtune_object(
    name: str, module: ModuleType, module_path: str, **kwargs
) -> Any:
    if name.startswith("_"):
        _raise_not_valid_object(name, module_path)

    try:
        obj = getattr(module, name)
    except AttributeError:
        _raise_not_valid_object(name, module_path)

    if isinstance(obj, ModuleType):
        _raise_not_valid_object(name, module_path)

    return obj(**kwargs)


def get_model(name: str, device: Union[str, torch.device], **kwargs) -> ModuleType:
    """Get known supported models by name"""
    with get_device(device):
        return _get_torchtune_object(name, models, "torchtune.models", **kwargs)


def get_tokenizer(name: str, **kwargs) -> Callable:
    """Get known supported tokenizers by name"""
    return _get_torchtune_object(name, tokenizers, "torchtune.tokenizers", **kwargs)


def get_dataset(name: str, **kwargs) -> Dataset:
    """Get known supported datasets by name"""
    return _get_torchtune_object(name, datasets, "torchtune.datasets", **kwargs)


def get_loss(loss: str) -> nn.Module:
    """Returns a loss function from torch.nn.

    Args:
        loss (str): name of the loss function.

    Returns:
        nn.Module: loss function.
    """
    return _get_torchtune_object(loss, torch.nn, "torch.nn")


def get_optimizer(optimizer: str, model: torch.nn.Module, lr: float) -> Optimizer:
    """Returns an optimizer function from torch.optim.

    Args:
        optimizer (str): name of the optimizer.
        model (torch.nn.Module): model to optimize.
        lr (float): learning rate.

    Returns:
        Optimizer: optimizer function.
    """
    return _get_torchtune_object(
        optimizer, torch.optim, "torch.optim", params=model.parameters(), lr=lr
    )


def get_lr_scheduler(
    lr_scheduler: str, optimizer: torch.optim.Optimizer, **kwargs
) -> LRScheduler:
    """Returns an optimizer function from torch.optim.

    Args:
        lr_scheduler (str): name of the learning rate scheduler.
        optimizer (torch.optim.Optimizer): optimizer.
        **kwargs: additional arguments to pass to the learning rate scheduler.

    Returns:
        LRScheduler: learning rate scheduler.

    Raises:
        ValueError: if the lr scheduler is not a valid scheduler from torch.optim.lr_scheduler or torchtune
    """
    try:
        return _get_torchtune_object(
            lr_scheduler,
            lr_schedulers,
            "torchtune.modules.lr_schedulers",
            optimizer=optimizer,
            **kwargs,
        )
    except ValueError:
        try:
            return _get_torchtune_object(
                lr_scheduler,
                torch.optim.lr_scheduler,
                "torch.optim.lr_scheduler",
                optimizer=optimizer,
                **kwargs,
            )
        except ValueError as e:
            raise ValueError(
                f"{lr_scheduler} is not a valid object from torch.optim.lr_scheduler or torchtune"
            ) from e


def get_metric_logger(metric_logger_type: str, **kwargs) -> "MetricLoggerInterface":
    """Get a metric logger based on provided arguments.

    Args:
        metric_logger_type (str): name of the metric logger, options are "wandb", "tensorboard", "stdout", "disk".
        **kwargs: additional arguments to pass to the metric logger

    Returns:
        MetricLoggerInterface: metric logger
    """
    return _get_torchtune_object(
        metric_logger_type,
        metric_logging,
        "torchtune.utils.metric_logging",
        **kwargs,
    )
