from torch.optim import Optimizer

__all__ = ["FusedOptimizerInBackward"]


class FusedOptimizerInBackward(Optimizer):
    """Fuses optimizer step into the backward pass for reducing memory usage.

    Args:
        base_optimizer_cls (Union[str, Optimizer]): The optimizer class to use for each parameter.
            Can be anything that fits the signature for :class:``torch.optim.Optimizer``.
        params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize.
        **kwargs: Additional arguments to pass to the base optimizer.

    Examples:
        >>> from torchtune.modules.optim import FusedOptimizerInBackward
        >>> from torch.optim import Adam
        >>> optimizer = FusedOptimizerInBackward(Adam, model.parameters(), lr=0.01)
        >>> optimizer.step()
    """

    def __init__(self, base_optimizer_cls, params, **kwargs):
        # Super hack to get this to work from a config :/
        if isinstance(base_optimizer_cls, str):
            base_optimizer_cls = eval(base_optimizer_cls)

        self.per_param_optimizers = {}
        self.param_to_opt = {}
        self._proxy_optimizer = None  # For use with LR schedulers
        self._param_groups = []

        for param in params:
            if not param.requires_grad:
                continue
            opt = base_optimizer_cls([param], **kwargs)
            self.per_param_optimizers[param] = opt
            self.param_to_opt[param] = opt
            self._param_groups.append(opt.param_groups[0])
            if self._proxy_optimizer is None:
                self._proxy_optimizer = opt

            # Hook to call .step() on this param's optimizer
            if hasattr(param, "register_post_accumulate_grad_hook"):
                param.register_post_accumulate_grad_hook(
                    lambda p=param: self._step_and_clear(p)
                )
            else:
                param.register_hook(lambda grad, p=param: self._step_and_clear(p))

    def _step_and_clear(self, param):
        opt = self.param_to_opt.get(param, None)
        if opt is None:
            return
        opt.step()
        param.grad = None

    def step(self, closure=None):
        # This is a no-op, we step on each param's optimizer in the hook
        if closure is not None:
            raise NotImplementedError(
                "FusedOptimizerInBackward does not support stepping via a closure."
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
    def param_groups(self):
        return (
            self._proxy_optimizer.param_groups
            if self._proxy_optimizer
            else self._param_groups
        )

    @property
    def defaults(self):
        return self._proxy_optimizer.defaults if self._proxy_optimizer else {}
