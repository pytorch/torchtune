# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from .attention import CausalSelfAttention  # noqa
from .feed_forward import FeedForward  # noqa
from .kv_cache import KVCache  # noqa
from .lr_schedulers import get_cosine_schedule_with_warmup  # noqa
from .position_embeddings import RotaryPositionalEmbeddings  # noqa
from .rms_norm import RMSNorm  # noqa
from .tokenizer import Tokenizer  # noqa
from .transformer import TransformerDecoder, TransformerDecoderLayer  # noqa

__all__ = [
    "CausalSelfAttention",
    "FeedForward",
    "get_cosine_schedule_with_warmup",
    "KVCache",
    "RotaryPositionalEmbeddings",
    "RMSNorm",
    "Tokenizer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
]


def get_loss(loss: str) -> nn.Module:
    """Returns a loss function from torch.nn.

    Args:
        loss (str): name of the loss function.

    Returns:
        nn.Module: loss function.

    Raises:
        ValueError: if the loss is not a valid loss from torch.nn.
    """
    try:
        return getattr(nn, loss)()
    except AttributeError as e:
        raise ValueError(f"{loss} is not a valid loss from torch.nn") from e


def get_optimizer(
    optimizer: str, model: torch.nn.Module, lr: float, weight_decay: float = 0.0
) -> Optimizer:
    """Returns an optimizer function from torch.optim.

    Args:
        optimizer (str): name of the optimizer.
        model (torch.nn.Module): model to optimize.
        lr (float): learning rate.
        weight_decay (float): weight decay for optimizer. Default is 0.0.

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
