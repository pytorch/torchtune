# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.optim.optimizer import Optimizer


def get_optimizer(optimizer: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0) -> Optimizer:
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
    try:
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        return getattr(torch.optim, optimizer)(
            trainable_params, lr=lr, weight_decay=weight_decay
        )
    except AttributeError as e:
        raise ValueError(
            f"{optimizer} is not a valid optimizer from torch.optim"
        ) from e


# TODO convert to folder when we support tuning specific optimizers
