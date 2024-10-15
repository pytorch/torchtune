# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Union

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtune.training.memory import OptimizerInBackwardWrapper


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(
    optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper]
) -> float:
    """
    Full_finetune_distributed and full_finetune_single_device assume all optimizers have
    the same LR, here to validate whether all the LR are the same and return if True.

    Args:
        optimizer (Union[torch.optim.Optimizer, OptimizerInBackwardWrapper]): A general
            optimizer input that could whether be a general optimizer or an optimizer
            warpper based on optimizer_in_backward.

    Returns:
        lr (float): The learning rate of the input optimizers.

    Raises:
        RuntimeError: If the learning rates of the input optimizer are not the same.
    """
    if isinstance(optimizer, OptimizerInBackwardWrapper):
        param_groups = []
        for param in optimizer.state_dict().values():
            param_groups.append(param["param_groups"][0])
    else:
        param_groups = optimizer.param_groups
    if len(param_groups) < 1:
        raise RuntimeError(
            f"Invalid optimizer param groups with len of: {len(param_groups)}"
        )

    # LR Schedulers are the same across all param groups for full_finetune right now
    lr = param_groups[0]["lr"]
    for group in param_groups:
        if group["lr"] != lr:
            raise RuntimeError("LR Schedulers are different across all param groups ")
    return lr
