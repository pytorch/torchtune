# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim.lr_scheduler import LRScheduler

from .cosine_with_warmup import get_cosine_schedule_with_warmup

__all__ = ["get_cosine_schedule_with_warmup"]

ALL_LR_SCHEDULERS = {"cosine_with_warmup": get_cosine_schedule_with_warmup}


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
        ValueError: if the lr scheduler is not a valid optimizer from torch.optim.
    """
    try:
        if lr_scheduler in ALL_LR_SCHEDULERS:
            return ALL_LR_SCHEDULERS[lr_scheduler](optimizer, **kwargs)
        else:
            getattr(torch.optim.lr_scheduler, lr_scheduler)(optimizer, **kwargs)
    except AttributeError as e:
        raise ValueError(
            f"{lr_scheduler} is not a valid learning rate scheduler from torch.optim.lr_scheduler or torchtune"
        ) from e
