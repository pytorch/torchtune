# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn


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


# TODO convert to folder when we support llm specific losses
