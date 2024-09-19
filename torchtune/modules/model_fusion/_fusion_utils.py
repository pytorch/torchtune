# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Dict, List

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


def get_fusion_params(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Return the subset of parameters from a model that correspond to fused
    modules. Assumes that any fusion class has defined the
    :func:`~torchtune.modules.model_fusion.FusionLayer.fusion_params` method.

    Args:
        model (nn.Module): Instance of model class containing some
            fusion params.

    Returns:
        Dict[str, nn.Parameter]: the subset of model's state dict containing
            only adapter parameters.

    """
    fusion_params = {}
    for k, v in model.named_modules():
        if hasattr(v, "fusion_params") and callable(v.fusion_params):
            current_fusion_params = v.fusion_params()
            for n, p in v.named_parameters(recurse=True):
                if n in current_fusion_params:
                    full_key = f"{k}.{n}" if k else n
                    fusion_params.update({full_key: p})
                    current_fusion_params.remove(n)
            assert (
                current_fusion_params == []
            ), f"Fusion params {current_adapter_params} not converted"
    return fusion_params
