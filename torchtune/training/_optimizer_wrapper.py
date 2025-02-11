from typing import Any, Dict, Optional

from omegaconf import DictConfig

from torch import nn
from torchtune import config, training
from torchtune.training.lr_schedulers import get_lr


class OptimizerWrapper:
    """
    Abstraction on optimizer to get rid of _optimizer_in_bwd checks.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg_optimizer: DictConfig,
        optimizer_in_bwd: bool = False,
        opt_state_dict: Optional[Dict[str, Any]] = None,
    ):
        self._optimizer_in_bwd = optimizer_in_bwd
        self._model = model

        if self._optimizer_in_bwd:
            self._optim_dict = {
                param: config.instantiate(cfg_optimizer, [param])
                for param in self._model.parameters()
            }
            training.register_optim_in_bwd_hooks(model=self._model, optim_dict=self._optim_dict)
            self._optim_ckpt_wrapper = training.create_optim_in_bwd_wrapper(
                model=self._model, optim_dict=self._optim_dict
            )
            if opt_state_dict is not None:
                for param in opt_state_dict.keys():
                    try:
                        training.load_from_full_optimizer_state_dict(
                            self._model,
                            self._optim_ckpt_wrapper.state_dict()[param],
                            opt_state_dict[param],
                            self._device,
                        )
                    except BaseException as e:
                        raise RuntimeError(
                            "Failed loading in-backward optimizer checkpoints."
                            "Please make sure run being restored from was using in-backward optimizer."
                        ) from e
        else:
            # Standard setup
            self._optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
            if opt_state_dict:
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    self._optimizer,
                    opt_state_dict,
                    self._device,
                )

    def zero_grad(self) -> None:
        if self._optimizer_in_bwd:
            for opt in self._optim_ckpt_wrapper.optim_map.values():
                opt.zero_grad()
        else:
            self._optimizer.zero_grad()

    def step(self) -> None:
        if self._optimizer_in_bwd:
            # No explicit step needed here.
            pass
        else:
            self._optimizer.step()

    def get_lr(self) -> float:
        if self._optimizer_in_bwd:
            # Return the learning rate of the first optimizer in the wrapper
            return get_lr(next(iter(self._optim_ckpt_wrapper.optim_map.values())))
        else:
            return get_lr(self._optimizer)
        
    def set_learning_rate_scheduler
        if self._optimizer_in_bwd:
            self._optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)


    def state_dict(self) -> Dict[str, Any]:
        if self._optimizer_in_bwd:
            return self._optim_ckpt_wrapper.state_dict()
        else:
            return self._optimizer.state_dict()
    
    def get_optimizer(self) -> tuple:
        if self._optimizer_in_bwd:
            return next(iter(self._optim_ckpt_wrapper.optim_map.values()))
        else:
            return self._optimizer

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load optimizer state dict.

        Args:
            state_dict (Dict[str, Any]): optimizer state_dict.
        """
        if self._optimizer_in_bwd:
            for param in state_dict.keys():
                training.load_from_full_optimizer_state_dict(
                    self._model,
                    self._optim_ckpt_wrapper.state_dict()[param],
                    state_dict[param],
                    self._device,
                )
        else:
            training.load_from_full_optimizer_state_dict(
                self._model,
                self._optimizer,
                state_dict,
                self._device,
            )
