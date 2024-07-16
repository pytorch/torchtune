# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging

from typing import Any, Callable, Dict, Set, Type, Union

import torch

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchtune.utils.logging import get_logger

_log: logging.Logger = get_logger()

ACWrapPolicyType: Type = Union[Set[Type], Callable[[nn.Module, bool, int], bool]]


def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy: ACWrapPolicyType, **kwargs
) -> None:
    """Utility to apply activation checkpointing to the passed-in model.

    Args:
        model (nn.Module): Model to apply activation checkpointing to.
        auto_wrap_policy (ACWrapPolicyType): Policy to wrap module.
            This can either be a set of ``nn.Module`` types, in which case, modules of the specified type(s)
            will be wrapped individually with activation checkpointing, or a ``callable`` policy describing
            how to wrap the model with activation checkpointing. For more information on authoring custom
            policies, please see this tutorial:
            https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html#transformer-wrapping-policy.
        **kwargs: additional arguments to pass to ``torch.distributed`` activation checkpointing.
    """
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)


def cleanup_before_training() -> None:
    """
    Call gc collect, empty CUDA cache, and reset peak memory stats.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class OptimizerInBackwardWrapper:
    """
    A bare-bones class meant for checkpoint save and load for optimizers running
    in backward. Usage is limited to the following:

    Note:
        This wrapper is only meant to be used for single-device use cases.
        Distributed use cases such as FSDP, which require specialized optimizer state checkpointing, are not supported.

    Args:
        optim_map (Dict[str, torch.optim.Optimizer]): Mapping from parameter names to optimizers.

    Example:
        >>> optim_dict = {
        >>>     p: config.instantiate(cfg_optimizer, [p])
        >>>     for p in self._model.parameters()
        >>> }
        >>>
        >>> # Save checkpoint
        >>> ckpt = OptimizerInBackwardWrapper(optim_dict).state_dict()
        >>> torch.save("/tmp/optim_ckpt", ckpt)
        >>>
        >>> # Load checkpoint
        >>> placeholder_optim_dict = {
        >>>     p: config.instantiate(cfg_optimizer, [p])
        >>>     for p in self._model.parameters()
        >>> }
        >>>
        >>> wrapper = OptimInBackwardWrapper(placeholder_optim_dict)
        >>>
        >>> # load_state_dict expects a dict produced by this class's
        >>> # state_dict method.
        >>> wrapper.load_state_dict(torch.load("/tmp/optim_ckpt"))
        >>> # placeholder_optim_dict now has updated optimizer states.

    """

    def __init__(self, optim_map: Dict[str, torch.optim.Optimizer]):
        self.optim_map = optim_map

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a state dict mapping parameter names to optimizer states. This
        state_dict is only loadable by this same class.

        Returns:
            Dict[str, Any]: state dict mapping parameter names to optimizer states.
        """
        return {p: opt.state_dict() for p, opt in self.optim_map.items()}

    def load_state_dict(self, optim_ckpt_map: Dict[str, Any]):
        """
        Load optimizer states from a state dict produced by this class's
        state_dict method.

        Args:
            optim_ckpt_map (Dict[str, Any]): state dict mapping parameter names to optimizer states.

        Raises:
            RuntimeError: If the optimizer state dict does not contain all the expected parameters.
        """
        params_covered = set()
        for param_name in optim_ckpt_map.keys():
            if param_name not in self.optim_map:
                raise RuntimeError(
                    f"Trying to load optimizer state for unexpected param {param_name}"
                )
            self.optim_map[param_name].load_state_dict(optim_ckpt_map[param_name])
            params_covered.add(param_name)
        # Ensure all params have been loaded into, report missing params
        missing_params = set(self.optim_map.keys()) - params_covered
        if missing_params:
            raise RuntimeError(
                f"Expected to load optimizer state for params {missing_params}!"
            )

    def get_optim_key(self, key: str) -> Any:
        """
        Returns value of key from an arbitrary optimizer running in backward. Note that
        this assumes all optimizer in backwards have the same value for the key, i.e.,
        are initialized with the same hyperparameters.
        """
        return list(self.optim_map.values())[0].param_groups[0][key]


def create_optim_in_bwd_wrapper(
    model: torch.nn.Module, optim_dict: Dict[torch.nn.Parameter, torch.optim.Optimizer]
) -> OptimizerInBackwardWrapper:
    """
    Create a wrapper for optimizer step running in backward.

    Args:
        model (torch.nn.Module): Model that contains parameters that are being optimized. For now,
            it is assumed that all parameters being optimized belong to a single top-level model.
            ``named_parameters`` attribute of ``model`` will be accessed to look up parameter names for
            parameters being optimized.
        optim_dict (Dict[torch.nn.Parameter, torch.optim.Optimizer]): Mapping from
            parameters to optimizers.

    Returns:
        ``OptimizerInBackwardWrapper``: Wrapper for optimizer states running in backward.
    """
    return OptimizerInBackwardWrapper(
        {n: optim_dict[p] for n, p in model.named_parameters()}
    )


def register_optim_in_bwd_hooks(
    model: torch.nn.Module, optim_dict: Dict[torch.nn.Parameter, torch.optim.Optimizer]
) -> None:
    """
    Register hooks for optimizer step running in backward.

    When fusing the optimizer step into backward, we need to call ``.step()`` on the optimizer
    for a given parameter as soon as its gradient is ready. This utility registers post-accumulate-grad
    hooks on all parameters in the model to achieve this.

    Args:
        model (torch.nn.Module): Model whose parameters will be optimized. Note that currently
            hooks for ALL parameters in the model will be registered.
        optim_dict (Dict[torch.nn.Parameter, torch.optim.Optimizer]): Mapping from
            parameters to optimizers.
    """

    def optim_step(param) -> None:
        optim_dict[param].step()
        optim_dict[param].zero_grad()

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optim_step)


def get_memory_stats(device: torch.device, reset_stats: bool = True) -> dict:
    """
    Computes a memory summary for the passed in device. If ``reset_stats`` is ``True``, this will
    also reset CUDA's peak memory tracking. This is useful to get data around relative use of peak
    memory (e.g. peak memory during model init, during forward, etc) and optimize memory for
    individual sections of training.

    Args:
        device (torch.device): Device to get memory summary for. Only CUDA devices are supported.
        reset_stats (bool): Whether to reset CUDA's peak memory tracking.

    Returns:
        Dict[str, float]: A dictionary containing the peak memory active, peak memory allocated,
        and peak memory reserved. This dict is useful for logging memory stats.

    Raises:
        ValueError: If the passed-in device is not CUDA.
    """
    if device.type != "cuda":
        raise ValueError(
            f"Logging memory stats is only supported on CUDA devices, got {device}"
        )

    peak_memory_active = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1e9
    peak_mem_alloc = torch.cuda.max_memory_allocated(device) / 1e9
    peak_mem_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    if reset_stats:
        torch.cuda.reset_peak_memory_stats(device)

    memory_stats = {
        "peak_memory_active": peak_memory_active,
        "peak_memory_alloc": peak_mem_alloc,
        "peak_memory_reserved": peak_mem_reserved,
    }
    return memory_stats


def log_memory_stats(stats: Dict[str, float]) -> None:
    """
    Logs a dict containing memory stats to the logger. ``stats`` should contain the fields
    ``peak_memory_active``, ``peak_memory_alloc``, and ``peak_memory_reserved`` as
    returned by :func:`torchtune.utils.get_memory_stats`.

    Args:
        stats (Dict[str, float]): A dictionary containing the peak memory active, peak memory
            allocated, and peak memory reserved stats.
    """
    _log.info(
        "Memory stats after model init:"
        f"\n\tGPU peak memory allocation: {stats['peak_memory_alloc']:.2f} GB"
        f"\n\tGPU peak memory reserved: {stats['peak_memory_reserved']:.2f} GB"
        f"\n\tGPU peak memory active: {stats['peak_memory_active']:.2f} GB"
    )
