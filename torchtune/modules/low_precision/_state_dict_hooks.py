import torch.nn as nn
from typing import Dict, Any
from torchao.dtypes.nf4tensor import NF4Tensor

def reparametrize_as_bf16_state_dict_post_hook(
    model: nn.Module,
    state_dict: Dict[str, Any],
    *args,
    offload_to_cpu: bool = True,
    **kwargs,
):
    """
    A state_dict hook that replaces nf4 tensors with their restored
    bf16 weight and optionally offloads the restored weight to CPU.

    This function is meant to be used with PyTorch's ``nn.Module._register_state_dict_hook``, i.e.
    >>> m = MyModule()
    >>> m._register_state_dict_hook(reparametrize_as_bf16_state_dict_post_hook)

    If the hook is registered per the above process, this hook will be called _after_ the module's
    ``state_dict`` method is called. The hook will replace all ``NF4Tensor`` instances by unquantizing
    them to the original dtype, and optionally offload the restored weight to CPU.

    Args:
        model (nn.Module): the model to take ``state_dict()`` on
        state_dict (Dict[str, Any]): the state dict to modify
        offload_to_cpu (bool): whether to offload the restored weight to CPU
    """
    for k, v in state_dict.items():
        if isinstance(v, NF4Tensor):
            state_dict[k] = v.get_original_weight()
            if offload_to_cpu:
                state_dict[k] = state_dict[k].cpu()
