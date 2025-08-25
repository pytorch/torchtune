# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, DTensor

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

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        params: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, params, muon_selector=None, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_lr=3e-4, adamw_betas=[0.95, 0.95], adamw_eps=1e-8, adamw_wd=0):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

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
        adamw_params = [p for n, p in named_params if not muon_selector(n, p)]

        super().__init__([*muon_params, *adamw_params], defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        # we cant pickle booleans for saving, so we will use 1=True, 0=False
        def assign_muon(p):
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = 1
            else:
                self.state[p]['use_muon'] = 0

        if isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group['params']:
                    assign_muon(p)
        else:
            for p in muon_params:
                assign_muon(p)

        def assign_adamw(p):
            # Do not use Muon for parameters in adamw_params
            self.state[p]['use_muon'] = 0

        if len(adamw_params) and isinstance(adamw_params[0], dict):
            for group in adamw_params:
                for p in group['params']:
                    assign_adamw(p)
        else:
            for p in adamw_params:
                assign_adamw(p)

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def to_dist(self, x, from_local=False, **meta):
        if from_local:
            return DTensor.from_local(
                x,
                device_mesh=meta["device_mesh"],
                placements=meta["placements"],
                shape=meta["shape"],
                stride=meta["stride"],
            )
        else:
            return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])


    def to_local(self, x, keep_sharded=False):
        if isinstance(x, DTensor):
            meta = dict(
                device_mesh=x.device_mesh,
                placements=x.placements,
                shape=x.shape,
                stride=x.stride(),
            )
            if keep_sharded:
                return x.to_local(), meta
            else:
                return x.full_tensor(), meta

        return x, None

    def zeropower_via_newtonschulz5(self, G, steps=10, eps=1e-7):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16()
        X /= (X.norm() + eps) # ensure top singular value <= 1
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            for i, p in enumerate(group['params']):
                if self.state[p]['use_muon'] == 1:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)

                    meta = None
                    if isinstance(g, DTensor):
                        g, meta = self.to_local(g, keep_sharded=False)
                    # gives NaNs when done with Dtensor, instead of throwing a typical op not supported error, quite sneaky
                    g = self.zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    if meta is not None:
                        g = self.to_dist(g, **meta)
                    g *= max(1, g.size(0)/g.size(1))**0.5

                    g = g.view_as(p.data).type_as(p.data)
                    p.data.add_(g, alpha=-lr)
                else:
                    # these are all pointwise so we can stay in Dtensor
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(g)
                        state['moment2'] = torch.zeros_like(g)
                    state['step'] += 1
                    step = state['step']
                    buf1 = state['moment1']
                    buf2 = state['moment2']
                    buf1.lerp_(g, 1-group['adamw_betas'][0])
                    buf2.lerp_(g.square(), 1-group['adamw_betas'][1])

                    g = buf1 / (group['adamw_eps'] + buf2.sqrt())

                    bias_correction1 = 1 - group['adamw_betas'][0]**step
                    bias_correction2 = 1 - group['adamw_betas'][1]**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * group['adamw_wd'])
                    p.data.add_(g, alpha=-lr/scale)