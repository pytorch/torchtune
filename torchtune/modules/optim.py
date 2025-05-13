# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, Type

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
        optimizer (Type[Optimizer]): The base optimizer class (e.g., AdamW).
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
        optimizer: Type[Optimizer],
        **optimizer_kwargs,
    ):
        # Initialize parent class with empty parameter groups
        super().__init__([], {})

        # Super hack to get this to work from a config :/
        if isinstance(optimizer, str):
            optimizer = eval(optimizer)

        self.per_param_optimizers = {}
        self.param_to_opt = {}
        self._proxy_optimizer = None  # For use with LR schedulers
        self._param_groups = []

        for param in params:
            if not param.requires_grad:
                continue
            opt = optimizer([param], **optimizer_kwargs)
            self.per_param_optimizers[param] = opt
            self.param_to_opt[param] = opt
            self._param_groups.append(opt.param_groups[0])
            if self._proxy_optimizer is None:
                self._proxy_optimizer = opt
            # Hook to call .step() on this param's optimizer
            param.register_post_accumulate_grad_hook(
                lambda p=param: self._step_and_clear(p)
            )

    def _step_and_clear(self, param):
        opt = self.param_to_opt.get(param, None)
        if opt is None:
            return
        opt.step()
        param.grad = None

    def step(self, closure=None) -> None:
        # This is a no-op, we step on each param's optimizer in the hook
        if closure is not None:
            raise RuntimeError(
                "OptimizerInBackward does not support stepping via a closure."
            )

    def zero_grad(self, set_to_none=True):
        for param, _ in self.per_param_optimizers.items():
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()

    def state_dict(self):
        return {
            "per_param": {
                id(p): opt.state_dict() for p, opt in self.per_param_optimizers.items()
            },
            "proxy": (
                self._proxy_optimizer.state_dict() if self._proxy_optimizer else {}
            ),
        }

    def load_state_dict(self, state_dict):
        for p, opt in self.per_param_optimizers.items():
            key = id(p)
            if key in state_dict.get("per_param", {}):
                opt.load_state_dict(state_dict["per_param"][key])
        if self._proxy_optimizer and "proxy" in state_dict:
            self._proxy_optimizer.load_state_dict(state_dict["proxy"])

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return (
            self._proxy_optimizer.param_groups
            if self._proxy_optimizer
            else self._param_groups
        )

    @property
    def defaults(self):
        return self._proxy_optimizer.defaults if self._proxy_optimizer else {}
