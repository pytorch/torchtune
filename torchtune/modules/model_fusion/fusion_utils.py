# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import List

from torch import nn


def register_fusion_module(module: nn.Module):
    """Add the method fusion_params to an nn.Module that
    marks all of the Modules parameters as fusion params.
    This can be used for a layer or an entire model that is
    added to combine two or more pretrained models.
    """

    def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        return [k for k, v in self.named_parameters()]

    module.fusion_params = functools.partial(fusion_params, module)
