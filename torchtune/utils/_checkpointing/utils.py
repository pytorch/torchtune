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
    force_override: bool = False,
):
    """
    Validates the state dict for checkpoint loading for a classifier model.
    To be used prior to a call to ``model.load_state_dict(state_dict)``.
    This function will overwrite the ``output.weight`` in the state-dict
    to be loaded with the ``output.weight`` in the model if the shapes
    for the ``output.weight`` do not match. You may also wish to override this behaviour,
    for example, if ``num_classes`` for your checkpoint and model are the same.

    Concretely, when fine-tuning a classifier model from the checkpoint of a base language model
    which has ``output.weight`` of shape ``[vocab_dim, embed_dim]``, we overwrite
    the ``output.weight`` in the state-dict to be loaded with the randomly initialized
    ``[num_classes, embed_dim]`` weight in the model. This is done in-place.

    Args:
        state_dict (Dict[str, torch.Tensor]): state dict to be loaded into the classifier model.
        model_named_parameters (Iterable[Tuple[str, torch.nn.Parameter]]): model named parameters
            from ``model.named_parameters()``.
        force_override (bool): Whether to replace ``output.weight`` in ``state_dict`` with the model's
            ``output.weight``, even if the shapes match.
    Notes:
        - ``output.bias`` will be ignored if present in ``state_dict``
        - This function will always replace the ``output.weight`` in ``state_dict``,
            if ``output.weight != model.output.weight``.

    Raises:
        AssertionError: if ``state_dict`` does not contain ``output.weight``.
        AssertionError: if ``model_named_parameters`` does not contain ``output.weight``.

    """
    output_weight = dict(model_named_parameters).get("output.weight", None)
    if "output.weight" not in state_dict:
        raise AssertionError(
            "Expected output.weight in state_dict, but it wasn't found."
        )
    if output_weight is None:
        raise AssertionError(
            "Expected output.weight in model_named_parameters, but it wasn't found."
        )
    if "output.bias" in state_dict:
        warn("Found output.bias in state dict - this will not be used!")
        state_dict.pop("output.bias")
    if state_dict["output.weight"].shape[0] != output_weight.shape[0] or force_override:
        state_dict["output.weight"] = output_weight
