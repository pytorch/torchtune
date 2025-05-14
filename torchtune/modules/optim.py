# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Union

import torch
from torch.optim import Optimizer

__all__ = ["OptimizerInBackward"]


class OptimizerInBackward(Optimizer):
    """Wraps a standard PyTorch optimizer class to perform parameter updates inside the backward pass.

    For each parameter, a separate optimizer instance is created, allowing in-place updates
    during gradient computation. This reduces peak memory usage by freeing gradients immediately
    after they are applied.

    Compatible with learning rate schedulers through delegation to a proxy optimizer instance.

    Args:
        params (Iterable[torch.nn.Parameter]): Model parameters to optimize.
        optimizer (Union[str, type[Optimizer]]): The base optimizer class (e.g., AdamW).
        **optimizer_kwargs: Additional arguments passed to the optimizer constructor.

    Example:
        >>> model = MyModel()
        >>> optimizer = OptimizerInBackward(model.parameters(), torch.optim.AdamW, lr=1e-3)
        >>> for input, target in data_loader:
        >>>     optimizer.zero_grad()
        >>>     output = model(input)
        >>>     loss = loss_fn(output, target)
        >>>     loss.backward()  # optimizer updates happen inside backward
        >>>     optimizer.step()  # no-op, but required for LR scheduler compatibility
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer: Union[str, type[Optimizer]],
        **optimizer_kwargs,
    ):
        # Super hack to get this to work from a config :/
        if isinstance(optimizer, str):
            optimizer: type[Optimizer] = eval(optimizer)

        self._per_param_optimizers = {}
        self._param_groups = []

        for param in params:
            if not param.requires_grad:
                continue
            opt = optimizer([param], **optimizer_kwargs)
            self._per_param_optimizers[param] = opt
            self._param_groups.append(opt.param_groups[0])
            # Hook to call .step() on this param's optimizer
            param.register_post_accumulate_grad_hook(
                lambda p=param: self._step_and_clear(p)
            )

        # Necessary to call this for setting up hooks, etc.
        super().__init__(self._param_groups, optimizer_kwargs)

    def _step_and_clear(self, param):
        opt = self._per_param_optimizers[param]
        opt.step()
        opt.zero_grad()

    def step(self, closure=None):
        # This is a no-op, we step on each param's optimizer in the hook
        if closure is not None:
            raise RuntimeError(
                "OptimizerInBackward does not support stepping via a closure."
            )

    def zero_grad(self, set_to_none=True):
        for opt in self._per_param_optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # Call the super() method in order to grab proper state and param groups
        state_dict = super().state_dict()
        state_dict["per_param_optimizers"] = {
            p: opt.state_dict() for p, opt in self._per_param_optimizers.items()
        }
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for p, opt in self._per_param_optimizers.items():
            opt.load_state_dict(state_dict["per_param_optimizers"][p])
