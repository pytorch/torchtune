# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Union

import torch

from torchtune.modules.common_utils import slice_str_to_array


class LayerDropout(torch.nn.Module):
    """
    A module that applies layer dropout to the input tensor of an underlying module.
    It drops a portion of an input tensor, applies the underlying module on the
    remaining parts of the tensor, and then concatenates with the dropped portion of the tensor.
    When applied during training, it can have a regularization effect, and can potentially speedup training.

    Args:
        prob (float): The probability of dropping an input. Defaults to 0.0.
        dim (Optional[int]): The dimension of input tensor along which to drop layers. Defaults to 0 (i.e., batch size).
        disable_on_eval (Optional[bool]): Whether to disable layer dropout during evaluation. Defaults to True.
        seed (Optional[int]): The seed for the random number generator. Defaults to None.
    Examples:
        >>> import torch
        >>> # Apply layer dropout to a lambda function
        >>> layer_dropout = LayerDropout(prob=0.5)
        >>> output = layer_dropout(lambda x: x**2, torch.randn(1))
        >>> # Apply layer dropout to a torch.nn.Linear module
        >>> linear = torch.nn.Linear(5, 3)
        >>> layer_dropout = LayerDropout(prob=0.5)
        >>> output = layer_dropout(linear, torch.randn(1, 5))
    """

    def __init__(
        self,
        prob: float = 0.0,
        dim: Optional[int] = 0,
        disable_on_eval: Optional[bool] = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.prob: float = prob
        self.dim = dim
        self.disable_on_eval: bool = disable_on_eval
        self.generator = torch.Generator(device="cpu")
        self.inferred: float = None

        if seed is not None:
            self.generator.manual_seed(seed)

    def forward(
        self,
        function: Union[Callable, torch.nn.Module],
        input: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply layer dropout to the input tensor.

        Args:
            function (Union[Callable, torch.nn.Module]): The function or module to apply to the input tensor.
            input (torch.Tensor): The input tensor.
            *args: Additional positional arguments passed to the function.
            **kwargs: Additional keyword arguments passed to the function.
        Returns:
            torch.Tensor: The output tensor after applying layer dropout.
        """
        n = input.shape[self.dim]

        if self.prob == 0 or (self.disable_on_eval and self.training is False):
            self.inferred = 1.0
            return function(input, *args, **kwargs)

        skip = (
            torch.bernoulli(torch.Tensor((n) * [self.prob]), generator=self.generator)
            .to(input.device)
            .to(input.dtype)
        )
        self.inferred = 1 - torch.mean(skip)
        ind_selected = (skip == 0).nonzero().squeeze()

        if ind_selected.numel() > 0:
            x_selected = torch.index_select(input, self.dim, ind_selected)
            out_selected = function(x_selected, *args, **kwargs)

        out = input.clone()
        assert (
            self.dim == 0
        ), "Currently only supporting dropping elements along the 0th dimension"
        if ind_selected.numel() > 0:
            out[ind_selected] = out_selected
        return out


class ModuleLayerDropoutWrapper(torch.nn.Module):
    """
    A wrapper module that adds layer dropout functionality to a given module.
    This class wraps a given module and applies layer dropout to it. It also
    provides getter and setter methods for the wrapped module's attributes.

    Args:
        module (torch.nn.Module): The module to wrap.
        dropout (LayerDropout): The layer dropout object.
    Examples:
        >>> import torch
        >>> from torch import nn
        >>> # Define a simple model
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(5, 3)
        ...         self.fc2 = nn.Linear(3, 2)
        ...
        ...     def forward(self, x):
        ...         return self.fc2(self.fc1(x))
        >>> model = MyModel()
        >>> fc1 = model.fc1
        >>> fc2 = model.fc2
        >>> # Apply layer dropout to the model
        >>> layer_dropout = LayerDropout(prob=0.5)
        >>> model = ModuleLayerDropoutWrapper(model, layer_dropout)
        >>> # Accessing attributes of the wrapped model
        >>> assert model.dropout.prob == 0.5
        >>> assert model.fc1 == fc1
        >>> assert model.fc2 == fc2
        >>> # Pass an input to the wrapped model as if you are passing it to the original model
        >>> output = model(torch.randn(1, 5))
    """

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
        """Return the state dictionary of the wrapped module."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load the state dictionary into the wrapped module."""
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


def get_scale(
    scale_type: ScaleType,
    scale_period: int,
    val: int,
) -> float:
    """
    Compute a scaling factor based on the provided scale type, period, and value.
    The scaling factor is designed to be 0 when the value is 0 and 1 when the value
    reaches or is larger than the scale period.

    Args:
        scale_type (ScaleType): The type of scaling to use.
        scale_period (int): The period over which the scaling factor increases from 0 to 1.
        val (int): The current value used to compute the scaling factor.
    Returns:
        float: The computed scaling factor.
    Examples:
        >>> get_scale(ScaleType.LINEAR, 10, 5)
        0.5
        >>> get_scale(ScaleType.LINEAR, 10, 0)
        0.0
        >>> get_scale(ScaleType.LOG, 10, 10)
        1.0
    """
    if scale_period == 0:
        return 1.0
    if val >= scale_period:
        return 1.0

    # all the equations below aim to make scale = 0 when val=0, and scale = 1 when val=scale_period
    scale = {
        ScaleType.UNIFORM: 1.0,
        ScaleType.EXP: math.pow(2, val / scale_period) - 1,
        ScaleType.LINEAR: val / scale_period,
        ScaleType.LOG: math.log(val + 1, scale_period + 1),
        ScaleType.SIN: math.sin(0.5 * math.pi * val / scale_period),
        ScaleType.SIGMOID: 1 / (1 + math.exp(-10 * (val / scale_period - 0.5))),
    }[scale_type]

    # ensure returned scale is between 0.0 and 1.0 (inclusive)
    return max(min(scale, 1.0), 0.0)


def prepare_layer_dropout(
    layers: Union[torch.nn.ModuleList, Iterable[torch.nn.Module]],
    prob_max: float = 0.0,
    prob_layer_scale: Optional[ScaleType] = ScaleType.UNIFORM,
    layers_str: Optional[str] = None,
    disable_on_eval: Optional[bool] = True,
) -> None:
    """
    Prepare a model's layers for layer dropout by wrapping each layer with a ModuleLayerDropoutWrapper.
    This function takes in a list of layers, the maximum probability of dropping a layer,
    the scaling type for the layer dropout probability, a string specifying which
    layers to apply dropout to, and a boolean indicating whether to disable dropout
    during evaluation. It then wraps each layer of the model inplace with a
    ModuleLayerDropoutWrapper, which applies layer dropout to the input tensor.

    Args:
        layers (Union[torch.nn.ModuleList, Iterable[torch.nn.Module]]): The list of layers to prepare for layer dropout.
        prob_max (float): The maximum probability of dropping a layer. Defaults to 0.0.
        prob_layer_scale (Optional[ScaleType]): The scaling type for the dropout probability across layers. Defaults to
            ScaleType.UNIFORM.
        layers_str (Optional[str]): A string specifying which layers to apply dropout to. Defaults to None which means
            apply to all layers.
        disable_on_eval (Optional[bool]): Whether to disable dropout during evaluation. Defaults to True.
    Returns:
        None
    Example:
        >>> import torch
        >>> from torch import nn
        >>> # Define a simple model
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.layers = nn.ModuleList([
        ...             nn.Linear(5, 3),
        ...             nn.Linear(3, 2),
        ...             nn.Linear(2, 1),
        ...             nn.Linear(1, 2),
        ...             nn.Linear(2, 3),
        ...         ])
        ...
        ...     def forward(self, x):
        ...         for layer in self.layers:
        ...             x = layer(x)
        ...         return x
        >>> model = MyModel()
        >>> # Apply layer dropout uniformly to all layers
        >>> prepare_layer_dropout(model.layers, prob_max=0.2, prob_layer_scale=ScaleType.UNIFORM)
        >>> # Apply layer dropout every other layer, as described in LayerDrop paper
            (Fan et al., https://arxiv.org/abs/1909.11556v1)
        >>> prepare_layer_dropout(model.layers, prob_max=0.2, prob_layer_scale=ScaleType.UNIFORM, layers_str="::2")
        >>> # Apply layer dropout that increases linearly across layers, as described in Progressive Layer
            Dropout paper (Zhang et al., https://arxiv.org/abs/2010.13369)
        >>> prepare_layer_dropout(model.layers, prob_max=0.2, prob_layer_scale=ScaleType.LINEAR)
        >>> # Apply layer dropout that increases exponentially across layers, as described in
            LayerSkip paper (Elhoushi et al., https://arxiv.org/abs/2404.16710)
        >>> prepare_layer_dropout(model.layers, prob_max=0.2, prob_layer_scale=ScaleType.EXP)
    """
    num_layers = len(layers)
    has_dropout = (
        slice_str_to_array(layers_str, num_layers)
        if layers_str
        else [True] * num_layers
    )
    for layer_id in range(len(layers)):
        prob = (
            prob_max
            * get_scale(
                scale_type=prob_layer_scale,
                scale_period=num_layers - 1,
                val=layer_id,
            )
            if has_dropout[layer_id]
            else 0.0
        )
        assert (
            prob >= 0.0 and prob <= prob_max
        ), f"prob={prob} should be between 0 and {prob_max}"
        # We would like each layer to have a different seed, so that we don't have the same samples skipped across layers.
        # Hence, we use the layer_id as a seed for each layer's dropout.
        layer_dropout = LayerDropout(
            prob, disable_on_eval=disable_on_eval, seed=layer_id
        )
        layers[layer_id] = ModuleLayerDropoutWrapper(layers[layer_id], layer_dropout)
