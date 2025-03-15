# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

import torch.nn as nn
from torchtune.config._utils import _get_component_from_path
from torchtune.modules.transformer import TransformerDecoder


def classifier_model(
    num_classes: int, base_model_path: str, **base_model_kwargs: Dict[str, Any]
) -> TransformerDecoder:
    """
    Create a classifier model from a base model by adapting the output layer.

    Args:
        num_classes (int): The number of classes for the classifier.
        base_model_path (str): The path to the base model builder, which
            must return an instance of ``TransformerDecoder``.
        **base_model_kwargs (Dict[str, Any]): Keyword arguments for the base model.

    Returns:
        TransformerDecoder: The base model, with the output layer adapted for the number of classes.

    Example:
        >>> from torchtune.models.common import classifier_model
        >>> model = classifier_model(num_classes=1, base_model_path="torchtune.models.llama3_2.llama3_2_1b")
        >>> model.output.weight.shape
        torch.Size([1, 4096])

    """
    model = _get_component_from_path(base_model_path)(**base_model_kwargs)

    if isinstance(model.output, nn.Linear):
        del model.output.weight
        if hasattr(model.output, "bias"):
            del model.output.bias
    model.output = nn.Linear(model.head_dim * model.num_heads, num_classes, bias=False)
    return model
