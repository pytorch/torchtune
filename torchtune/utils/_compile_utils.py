# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


_TORCH_COMPILE_WRAPPER_PREFIX = "_orig_mod."


def wrap_compile(model: nn.Module) -> None:
    """
    Wraps the model with torch.compile. This function will also
        register a state_dict post hook that allows state_dicts produced
        with torch.compile training to behave as regular eager mode models.
        In particular, it strips away a torch.compile specific prefix
        added to the state_dict by torch.compile.

        Args:
                model (nn.Module): model to wrap with compile.

        Returns:
            None
    """
    model = torch.compile(model)
    model._register_state_dict_hook(_remove_torch_compile_prefix)
    return model


def _remove_torch_compile_prefix(model, state_dict, *args, **kwargs):
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(_TORCH_COMPILE_WRAPPER_PREFIX):
            newkey = key[len(_TORCH_COMPILE_WRAPPER_PREFIX) :]
            state_dict[newkey] = state_dict.pop(key)
