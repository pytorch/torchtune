# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torchtune.training import (
    create_optim_in_bwd_wrapper,
    register_optim_in_bwd_hooks,
    set_activation_checkpointing,
)


class TestSetActivationCheckpointing:
    @pytest.fixture
    def model(self) -> int:
        return nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def _verify(self, model):
        for submodule in model.modules():
            if isinstance(submodule, CheckpointWrapper):
                assert isinstance(submodule._checkpoint_wrapped_module, nn.Linear)

    def test_activation_checkpoint_set_policy(self, model):
        set_activation_checkpointing(model=model, auto_wrap_policy={nn.Linear})
        self._verify(model)

    def test_activation_checkpoint_custom_policy(self, model):
        def custom_policy(module: nn.Module, recurse: bool, **kwargs) -> bool:
            if recurse:
                return True
            return isinstance(module, nn.Linear)

        set_activation_checkpointing(model=model, auto_wrap_policy=custom_policy)
        self._verify(model)


def _run_dummy_step(model, wrapper):
    with torch.no_grad():
        for p in model.parameters():
            p.grad = torch.rand_like(p)
    for v in wrapper.optim_map.values():
        v.step()
        v.zero_grad()


def _validate_dicts(d1, d2):
    if len(d1) != len(d2):
        return False
    for k, v in d1.items():
        if k not in d2:
            return False
        if isinstance(v, dict):
            return _validate_dicts(v, d2[k])
        else:
            if isinstance(v, torch.Tensor):
                if not torch.allclose(v, d2[k]):
                    return False
            elif v != d2[k]:
                return False
    return True


@pytest.fixture
def model():
    return torch.nn.Linear(10, 1)


@pytest.fixture
def optim_dict(model):
    return {p: torch.optim.AdamW([p], lr=0.01) for p in model.parameters()}


@pytest.fixture
def wrapper(model, optim_dict):
    return create_optim_in_bwd_wrapper(model, optim_dict)


class TestOptimInBackward:
    def test_state_dict_save_load(self, model, wrapper):
        # Run a dummy step to create optimizer states
        _run_dummy_step(model, wrapper)

        sd = wrapper.state_dict()
        new_optim_dict = create_optim_in_bwd_wrapper(
            model, {p: torch.optim.AdamW([p], lr=0.01) for p in model.parameters()}
        )
        assert not _validate_dicts(sd, new_optim_dict.state_dict())
        new_optim_dict.load_state_dict(sd)
        assert _validate_dicts(sd, new_optim_dict.state_dict())

    def test_missing_unexpected_param_load_raises(self, model, wrapper):
        # Run a dummy step to create optimizer states
        _run_dummy_step(model, wrapper)
        sd = wrapper.state_dict()
        new_optim_dict = create_optim_in_bwd_wrapper(
            model, {p: torch.optim.AdamW([p], lr=0.01) for p in model.parameters()}
        )
        with pytest.raises(RuntimeError, match="Expected to load optimizer state"):
            sd.pop(next(iter(sd.keys())))
            new_optim_dict.load_state_dict(sd)

        sd = wrapper.state_dict()
        sd["new_key"] = 1234
        with pytest.raises(RuntimeError, match="unexpected param"):
            new_optim_dict.load_state_dict(sd)


class TestRegisterOptimHooks:
    def test_register_optim_in_bwd_hooks(self, model, optim_dict):
        register_optim_in_bwd_hooks(model, optim_dict)
        # Ensure backward() updates the parameters and sets grads to None
        orig_params = [p.clone().detach() for p in model.parameters()]
        model(torch.rand(2, 10)).sum().backward()
        for p, orig_p in zip(model.parameters(), orig_params):
            assert not p.grad
            assert not torch.allclose(p, orig_p)
