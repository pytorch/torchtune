# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch
from torch.optim import Optimizer

__all__ = ["OptimizerInBackward"]


class OptimizerInBackward(Optimizer):
    """Wraps a standard PyTorch optimizer class to perform parameter updates inside the backward pass.

    For each parameter, a separate optimizer instance is created, allowing in-place updates
    during gradient computation. This reduces peak memory usage by freeing gradients immediately
    after they are applied.

    Args:
        params (Iterable[torch.nn.Parameter]): Model parameters to optimize.
        optimizer_cls (type[Optimizer]): The optimizer class to use in backward pass (e.g., AdamW).
        **optimizer_kwargs: Additional arguments passed to the optimizer constructor.

    Example:
        >>> model = MyModel()
        >>> optimizer = OptimizerInBackward(model.parameters(), torch.optim.AdamW, lr=1e-3)
        >>> for input, target in dataloader:
        >>>     optimizer.zero_grad()
        >>>     output = model(input)
        >>>     loss = loss_fn(output, target)
        >>>     loss.backward()  # optimizer updates happen inside backward
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        optimizer_cls: type[Optimizer],
        **optimizer_kwargs,
    ):
        params = list(params)  # Make a copy of params
        self._optimizers = {}
        self._param_to_index = {}

        for idx, p in enumerate(params):
            self._param_to_index[p] = idx
            if p.requires_grad:
                self._optimizers[idx] = optimizer_cls([p], **optimizer_kwargs)
                p.register_post_accumulate_grad_hook(
                    lambda param=p: self._step_and_clear(param)
                )

        super().__init__(params, optimizer_kwargs)

    def _step_and_clear(self, param):
        idx = self._param_to_index[param]
        self._optimizers[idx].step()
        self._optimizers[idx].zero_grad()

    def step(self, closure=None):
        if closure is not None:
            raise RuntimeError("OptimizerInBackward does not support closures.")

    def zero_grad(self, set_to_none=True):
        for opt in self._optimizers.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "param_groups": super().state_dict()["param_groups"],
            "optimizers": {
                str(idx): opt.state_dict() for idx, opt in self._optimizers.items()
            },
            "state": {},  # State is handled by individual optimizers
        }

    def load_state_dict(self, state_dict):
        super().load_state_dict(
            {"param_groups": state_dict["param_groups"], "state": {}}
        )
        for idx, opt in self._optimizers.items():
            opt.load_state_dict(state_dict["optimizers"][str(idx)])
