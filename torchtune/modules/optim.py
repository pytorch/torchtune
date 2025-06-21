# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch
from torch.optim import Optimizer
import torch.distributed as dist

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

######################################################
#
#   This code is referred from https://github.com/KellerJordan/Muon repo.
#   @misc{jordan2024muon,
#     author       = {Keller Jordan and Yuchen Jin and Vlado Boza and You Jiacheng and
#                     Franz Cesista and Laker Newhouse and Jeremy Bernstein},
#     title        = {Muon: An optimizer for hidden layers in neural networks},
#     year         = {2024},
#     url          = {https://kellerjordan.github.io/posts/muon/}
#   }
#   Changes have been made wherever necessary.
#
######################################################

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class SingleDeviceMuonWithAuxAdam(Optimizer):
    def __init__(
        self,
        params,     # Pass model.named_parameters()
        *,
        muon_selector=None,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        adam_lr: float = 3e-4,
        adam_betas=(0.9, 0.95),
        adam_eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):        
        if muon_selector is None:
            muon_selector = lambda name, param: (
                param.requires_grad and
                param.ndim >= 2 and                 # Check if scalar
                "embed" not in name.lower() and     # Check if embedding layer
                "tok" not in name.lower() and       # Check if token embeddings
                "head" not in name.lower() and      # Check if output head
                "bias" not in name.lower()          # Check if bias term
            )

        named_params = list(params)

        muon_params = [p for n, p in named_params if muon_selector(n, p)]
        adam_params = [p for n, p in named_params if not muon_selector(n, p)]

        muon_params.sort(key=lambda p: p.size(), reverse=True)

        super().__init__(
            [
                dict(params=muon_params,
                     lr=muon_lr,
                     momentum=muon_momentum,
                     weight_decay=weight_decay,
                     use_muon=True),
                dict(params=adam_params,
                     lr=adam_lr,
                     betas=adam_betas,
                     eps=adam_eps,
                     weight_decay=weight_decay,
                     use_muon=False),
            ],
            defaults={}
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,     # Pass model.named_parameters()
        *,
        muon_selector=None,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        adam_lr: float = 3e-4,
        adam_betas=(0.9, 0.95),
        adam_eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):        
        if muon_selector is None:
            muon_selector = lambda name, param: (
                param.requires_grad and
                param.ndim >= 2 and                 # Check if scalar
                "embed" not in name.lower() and     # Check if embedding layer
                "tok" not in name.lower() and       # Check if token embeddings
                "head" not in name.lower() and      # Check if output head
                "bias" not in name.lower()          # Check if bias term
            )

        named_params = list(params)

        muon_params = [p for n, p in named_params if muon_selector(n, p)]
        adam_params = [p for n, p in named_params if not muon_selector(n, p)]

        muon_params.sort(key=lambda p: p.size(), reverse=True)

        super().__init__(
            [
                dict(params=muon_params,
                     lr=muon_lr,
                     momentum=muon_momentum,
                     weight_decay=weight_decay,
                     use_muon=True),
                dict(params=adam_params,
                     lr=adam_lr,
                     betas=adam_betas,
                     eps=adam_eps,
                     weight_decay=weight_decay,
                     use_muon=False),
            ],
            defaults={}
        )

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])