# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import ModuleType

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from torchtune import datasets, models


def _raise_not_valid_object(name: str, module_path: str) -> None:
    raise ValueError(f"{name} is not a valid object from {module_path}")


def _get_torchtune_object(
    name: str, module: ModuleType, module_path: str, **kwargs
) -> Any:
    if name.startswith("_"):
        _raise_not_valid_object(name, module_path)

    try:
        obj = getattr(module, name)
    except AttributeError as e:
        _raise_not_valid_object(name, module_path)

    if isinstance(obj, ModuleType):
        _raise_not_valid_object(name, module_path)

    return obj(**kwargs)


def get_model(name: str, device: Union[str, torch.device], **kwargs) -> Module:
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

    Raises:
        ValueError: if the loss is not a valid loss from torch.nn.
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

    Raises:
        ValueError: if the optimizer is not a valid optimizer from torch.optim.
    """
    return _get_torchtune_object(
        optimizer, torch.optim, "torch.optim", params=model.parameters(), lr=lr
    )
