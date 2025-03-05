# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torchtune.config._utils import _get_component_from_path
from typing import Dict, Any
import torch

from torchtune.modules.tied_linear import TiedLinear


def classifier_model(num_classes: int, base_model: str, **base_model_kwargs: Dict[str, Any]) -> nn.Module:
    """
    Create a classifier model from a base model by adapting the output layer.

    Args:
        num_classes (int): The number of classes for the classifier.
        base_model_path (str): The path to the base model.
        base_model_kwargs (Dict[str, Any]): Keyword arguments for the base model.

    Returns:
        nn.Module: The classifier model.

    Example:

    """
    model = _get_component_from_path(base_model)(**base_model_kwargs)

    if isinstance(model.output, nn.Linear):
        del model.output.weight
        if hasattr(model.output, "bias"):
            del model.output.bias
    model.output = nn.Linear(model.head_dim * model.num_heads, num_classes, bias=False)
    print(model.output)
    print(model.output.weight)
    # import pdb
    # pdb.set_trace()
    return model
    model.output = nn.Linear(model.head_dim * model.num_heads, num_classes, bias=False)

    # if isinstance(model.output, TiedLinear):
    #     del model.output.tied_module.weight
    # else:
    #     del model.output.weight
    # model.output.register_parameter("weight", nn.Parameter(torch.empty(num_classes, model.head_dim * model.num_heads)))
    # # init weight with kaiming uniform
    # if hasattr(model.output, "bias"):
    #     del model.output.bias
    #     model.output.register_parameter("bias", nn.Parameter(torch.empty(num_classes)))
    #     # init bias with zeros
    # model.output.reset_parameters()
    # # model.output = nn.Linear(model.head_dim * model.num_heads, num_classes, bias=False)
    return model
