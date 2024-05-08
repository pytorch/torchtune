# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torch import nn


def wrap_compile(model: nn.Module) -> nn.Module:
    """
    Wraps the model with torch.compile.

    Args:
        model (nn.Module): model to wrap with compile.

    Returns:
        nn.Module: wrapped model
    """
    # TORCH_COMPILE_BACKEND can be set as an env var to override default torch.compile backend.
    # Currently only used in unittesting to work around https://github.com/pytorch/torchtune/issues/676
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    model.compile(backend=backend)
    return model
