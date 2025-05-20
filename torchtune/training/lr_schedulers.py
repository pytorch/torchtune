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
    min_lr_warmup: float = 0.0,
    min_lr_decay: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    ``min_lr_warmup`` to ``lr`` over ``num_warmup_steps``, then decreases to
    ``min_lr_decay`` on a cosine schedule over the remaining
    ``num_training_steps - num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to min_lr_ratio_decay following a half-cosine).
        min_lr_warmup (float): Minimum learning rate during warmup phase. Defaults to 0.0
        min_lr_decay (float): Minimum learning rate during decay phase. Defaults to 0.0
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """
    if min_lr_warmup > 0.0 or min_lr_decay > 0.0:
        lr = get_lr(optimizer)
        min_lr_ratio_decay = min_lr_decay / lr
        min_lr_ratio_warmup = min_lr_warmup / lr
    else:
        min_lr_ratio_decay = 0.0
        min_lr_ratio_warmup = 0.0

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            warmup_ratio = current_step / max(1, num_warmup_steps)
            return min_lr_ratio_warmup + (1 - min_lr_ratio_warmup) * warmup_ratio

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        decay_ratio = (1 - min_lr_ratio_decay) * cosine_lr_multiple + min_lr_ratio_decay
        return max(0.0, decay_ratio)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(
    optimizer: Union[torch.optim.Optimizer, OptimizerInBackwardWrapper],
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
