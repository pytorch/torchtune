import pytest
import torch
from torch.optim import SGD
from your_module import OptimizerInBackwardWrapper, create_optim_in_bwd_wrapper, register_optim_in_bwd_hooks
@pytest.fixture
def model():
    return torch.nn.Linear(10, 1)
@pytest.fixture
def optim_dict(model):
    return {p: SGD([p], lr=0.01) for p in model.parameters()}
@pytest.fixture
def wrapper(optim_dict):
    return OptimizerInBackwardWrapper(optim_dict)
def test_state_dict(wrapper, optim_dict):
    state_dict = wrapper.state_dict()
    assert isinstance(state_dict, dict)
    assert len(state_dict) == len(optim_dict)
def test_load_state_dict(wrapper, optim_dict):
    state_dict = wrapper.state_dict()
    wrapper.load_state_dict(state_dict)
    for param_name, optimizer in optim_dict.items():
        assert optimizer.state_dict() == state_dict[param_name]
def test_get_optim_key(wrapper):
    lr = wrapper.get_optim_key('lr')
    assert lr == 0.01
def test_create_optim_in_bwd_wrapper(model, optim_dict):
    wrapper = create_optim_in_bwd_wrapper(model, optim_dict)
    assert isinstance(wrapper, OptimizerInBackwardWrapper)
def test_register_optim_in_bwd_hooks(model, optim_dict):
    register_optim_in_bwd_hooks(model, optim_dict)
    # Here, you would need to add assertions based on what you expect to happen
    # when the hooks are registered. This could involve running a forward and
    # backward pass and checking the gradients, for example.



class TestOptimUtils:
    def test_optim_in_bwd_wrapper(self):
        pass
