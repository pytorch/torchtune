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

    For example, you might want to add a projection head
    head onto an encoder to learn a projection from the
    pre-trained encodings to the decoder's embedding space. This
    is typical with both Deep Fusion and Early Fusion models.

    Example:
        >>> projection_head = FeedForward(...)
        >>> register_fusion_module(projection_head))
        >>> encoder = nn.Sequential(clip_vit_224(), projection_head)

    Args:
        module (nn.Module): module to add the fusion_params method to
    """

    def fusion_params(self) -> List[str]:
        """
        Return parameters of fusion layer.
        """
        return [k for k, v in self.named_parameters()]

    module.fusion_params = functools.partial(fusion_params, module)
