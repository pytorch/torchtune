# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union

import torch.nn as nn
from torchtune.config._utils import _get_component_from_path
from torchtune.modules.transformer import TransformerDecoder

# TODO (SalmanMohammadi) - add a tutorial for fine-tuning classifiers
def classifier_model(
    num_classes: int, base_model_path: str, **base_model_kwargs: Dict[str, Any]
) -> Union[TransformerDecoder, nn.Module]:
    """
    Create a classifier model from a base model by adapting the output layer.

    Note:
        This builder does not support models which apply PEFT to the output layer.

    Args:
        num_classes (int): The number of classes for the classifier.
        base_model_path (str): The path to the base model builder, which
            must return an instance of ``TransformerDecoder``, or a model with
            a ``decoder`` attribute that is an instance of ``TransformerDecoder``.
        **base_model_kwargs (Dict[str, Any]): Keyword arguments for the base model.

    Returns:
        Union[TransformerDecoder, nn.Module]: The base model, with the output layer adapted for the number of classes.

    Raises:
        ValueError: If the base model does not have a valid output layer to adapt.

    Example:
        >>> from torchtune.modules import classifier_model
        >>> model = classifier_model(num_classes=1, base_model_path="torchtune.models.llama3_2.llama3_2_1b")
        >>> model.output.weight.shape
        torch.Size([1, 4096])

    """
    model = _get_component_from_path(base_model_path)(**base_model_kwargs)

    if hasattr(model, "output"):
        del model.output
        model.output = nn.Linear(
            model.head_dim * model.num_heads, num_classes, bias=False
        )
    elif hasattr(model, "decoder") and hasattr(model.decoder, "output"):
        del model.decoder.output
        model.decoder.output = nn.Linear(
            model.decoder.head_dim * model.decoder.num_heads, num_classes, bias=False
        )
    else:
        raise ValueError("Could not find a valid output layer to adapt.")
    return model
