# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Callable, Optional
import math
import torch

from torchtune.modules.common_utils import slice_str_to_array

class LayerDropout(torch.nn.Module):
    def __init__(self, prob=0.0, dim=0, disable_on_eval=True, seed=None):
        super().__init__()
        self.prob: float = prob
        self.dim = dim
        self.disable_on_eval: bool = disable_on_eval
        self.generator = torch.Generator(device="cpu")
        self.inferred: float = None

        if seed is not None:
            self.generator.manual_seed(seed)

    def forward(self, function: Callable, input: torch.Tensor, *args, **kwargs):
        n = input.shape[self.dim]

        if self.prob == 0 or (self.disable_on_eval and self.training is False):
            self.inferred = 1.0
            return function(input, *args, **kwargs)

        skip = torch.bernoulli(torch.Tensor((n) * [self.prob]), generator=self.generator).to(input.device).to(input.dtype)
        self.inferred = 1 - torch.mean(skip)
        ind_selected = (skip == 0).nonzero().squeeze()

        if ind_selected.numel() > 0:
            x_selected = torch.index_select(input, self.dim, ind_selected)
            out_selected = function(x_selected, *args, **kwargs)

        out = input.clone()
        assert self.dim == 0, "Currently only supporting dropping elements along the 0th dimension"
        if ind_selected.numel() > 0:
            out[ind_selected] = out_selected
        return out

class ModuleLayerDropoutWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, dropout: LayerDropout):
        super().__init__()
        self.module = module
        self.dropout = dropout

    def forward(self, input: torch.Tensor, *args, **kwargs):
        return self.dropout(self.module, input, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    def __setattr__(self, name: str, value: Any) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__setattr__(name, value)  # defer to nn.Module's logic
        except AttributeError:
            return setattr(self.module, name, value)  # fallback to wrapped module

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.module.load_state_dict(state_dict, *args, **kwargs)
        return

class ScaleType(str, Enum):
    UNIFORM = "uniform"
    EXP = "exp"
    LINEAR = "linear"
    LOG = "log"
    SIN = "sin"
    SIGMOID = "sigmoid"
    STEP = "step"

def get_scale(scale_type: ScaleType, scale_period: int, val: int):
    if scale_period == 0:
        return 1

    # all the equations below aim to make scale = 0 when val=0, and scale = 1 when val=scale_period
    return {
        ScaleType.UNIFORM: 1,
        ScaleType.EXP: math.exp(val * math.log(2) / scale_period) - 1,
        ScaleType.LINEAR: val / scale_period,
        ScaleType.LOG: math.log(val + 1) / math.log(scale_period + 1),
        ScaleType.SIN: math.sin(0.5 * math.pi * val / scale_period),
        ScaleType.SIGMOID: 1 / (1 + math.exp(-10 * (val / scale_period - 0.5))),
    }[scale_type]

# TODO: rename to prepare() just like quantizer()?
def apply_layer_dropout_modules(model, prob_max: float= 0.0, prob_layer_scale: ScaleType = ScaleType.EXP, layers_str: Optional[str] = None, disable_on_eval: bool = True):
    num_layers = len(model.layers)
    has_dropout = slice_str_to_array(layers_str, num_layers) if layers_str else [True] * num_layers
    for layer_id in range(len(model.layers)):
        prob = prob_max * get_scale(
            scale_type = prob_layer_scale,
            scale_period = num_layers - 1,
            val = layer_id,
        ) if has_dropout[layer_id] else 0.0
        assert prob >= 0.0 and prob <= prob_max, f"prob={prob} should be between 0 and {prob_max}"
        # We would like each layer to have a different seed, so that we don't have the same samples skipped across layers. Hence, we use the layer_id as a seed for each layer's dropout.
        layer_dropout = LayerDropout(prob, disable_on_eval=disable_on_eval, seed=layer_id)
        model.layers[layer_id] = ModuleLayerDropoutWrapper(model.layers[layer_id], layer_dropout)
