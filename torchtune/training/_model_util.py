# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torch


def disable_dropout(model: torch.nn.Module) -> None:
    """
    Disables dropout layers in the given model.

    Args:
        model (torch.nn.Module): The model in which dropout layers should be disabled.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            warnings.warn(
                f"Found Dropout with value {module.p} in module {module}. Setting to zero."
            )
            module.p = 0
