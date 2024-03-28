import pytest
import torch
from torchtune.utils import (
    create_optim_in_bwd_wrapper,
    OptimizerInBackwardWrapper,
    register_optim_in_bwd_hooks,
)


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
