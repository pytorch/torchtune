# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union

import torch.nn as nn
from torchtune.config._utils import _get_component_from_path
from torchtune.models.llama3_2_vision._model_builders import DeepFusionModel
from torchtune.modules.transformer import TransformerDecoder


def classifier_model(
    num_classes: int, base_model_path: str, **base_model_kwargs: Dict[str, Any]
) -> Union[TransformerDecoder, DeepFusionModel, nn.Module]:
    """
    Create a classifier model from a base model by adapting the output layer.

    Args:
        num_classes (int): The number of classes for the classifier.
        base_model_path (str): The path to the base model builder, which
            must return an instance of ``TransformerDecoder``, or a model with
            a ``decoder`` attribute that is an instance of ``TransformerDecoder``.
        **base_model_kwargs (Dict[str, Any]): Keyword arguments for the base model.

    Returns:
        Union[TransformerDecoder, DeepFusionModel, nn.Module]: The base model, with
            the output layer adapted for the number of classes.

    Raises:
        ValueError: If the base model does not have a valid output layer to adapt.

    Example:
        >>> from torchtune.models.common import classifier_model
        >>> model = classifier_model(num_classes=1, base_model_path="torchtune.models.llama3_2.llama3_2_1b")
        >>> model.output.weight.shape
        torch.Size([1, 4096])

    """
    model = _get_component_from_path(base_model_path)(**base_model_kwargs)

    if hasattr(model, "output"):
        del model.output
        head_dim = model.head_dim
        num_heads = model.num_heads
    elif hasattr(model, "decoder") and hasattr(model.decoder, "output"):
        del model.decoder.output
        head_dim = model.decoder.head_dim
        num_heads = model.decoder.num_heads
    else:
        raise ValueError("Could not find a valid output layer to adapt.")
    model.output = nn.Linear(head_dim * num_heads, num_classes, bias=False)
    return model
