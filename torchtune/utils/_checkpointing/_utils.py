# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, Tuple
from warnings import warn

import torch


def update_state_dict_for_classifier(
    state_dict: Dict[str, torch.Tensor],
    model_named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
):
    """
    Validates the state dict for checkpoint loading for a classifier model, to be
    used prior to a call to ``model.load_state_dict(state_dict)``.
    When fine-tuning a classifier model from the checkpoint of a base language model
    which has ``output.weight`` of shape ``[vocab_dim, embed_dim]``, we overwrite
    the ``output.weight`` in the state-dict to be loaded with the randomly initialized
    weight in the model. This is done in-place.

    When fine-tuning a classifier model from the checkpoint of a base classifier model,
    this function mostly defers to ``model.load_state_dict`` for validation, thus it is
    reccomended to call this when ``not self._resume_from_checkpoint`` in a recipe.

    Args:
        state_dict (Dict[str, torch.Tensor]): state dict to be loaded into the classifier model.
        model_named_parameters (Iterable[Tuple[str, torch.nn.Parameter]]): model named parameters
            from ``model.named_parameters()``.

    Raises:
        AssertionError: if ``state_dict`` does not contain ``output.weight``.
    """
    output_weight = [
        (k, v) for (k, v) in model_named_parameters if k == "output.weight"
    ]
    if "output.weight" not in state_dict:
        raise AssertionError(
            "Expected output.weight in state_dict, but it wasn't found."
        )
    if "output.bias" in state_dict:
        warn("Found 'output.bias' in state dict - this will not be used!")
        state_dict.pop("output.bias")
    if state_dict["output.weight"].shape != output_weight[0][1].shape:
        warn(
            f"Found output.weight with {state_dict['output.weight'].shape} "
            f"in checkpoint. This will be overwritten with model's output.weight "
            f"with {output_weight[0][1].shape}"
        )
        state_dict["output.weight"] = output_weight[0][1]
