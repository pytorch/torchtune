import torch

import torch.nn as nn
from torch import Tensor
from torchao.dtypes.nf4tensor import linear_nf4, NF4Tensor
from typing import Optional, Dict, Any


class FrozenNF4Linear(nn.Linear):
    """
    A linear layer similar to ``torch.nn.Linear`` but uses a quantized
    NF4Tensor as its weight. This class also freezes its ``weight`` parameter
    and is meant to be used as the base Linear layer for modeling
    use cases such as QLoRA where base model parameters are frozen.
    NOTE: biases are currently not supported.
    NOTE: This class always creates the underlying full precision weight as bf16 dtypte. Note that
        this will override the default PyTorch dtype that is set via `torch.set_default_dtype`.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        device (Optional[torch.device]): device to use for the underlying weight. If ``None``, uses the default
            device given by `torch.get_default_device()`.
        kwargs: Dict[str, Any]: any additional arguments to pass to the underlying Linear layer.
    """

    def __init__(
        self, in_dim: int, out_dim: int, device: Optional[torch.device] = None, **kwargs
    ):
        if "bias" in kwargs and kwargs.pop("bias"):
            raise RuntimeError("FrozenNF4Linear does not currently support biases!")

        if "dtype" in kwargs:
            kwargs_dtype = kwargs.pop("dtype")
            if kwargs_dtype != torch.bfloat16:
                raise RuntimeError(
                    "FrozenNF4Linear is only supported with bf16 parameter currently."
                )
        super().__init__(in_dim, out_dim, device=device, dtype=torch.bfloat16, **kwargs)
        self.weight.requires_grad_(False)

        self.nf4_weight = NF4Tensor.from_tensor(self.weight.data)
        # re-register self.weight as the nf4 weight, so that the nf4 weight
        # shows up as expected in .parameters, state_dict, etc.
        self.weight = torch.nn.Parameter(self.nf4_weight, requires_grad=False)

        # TODO: likely need to handle state_dict save & load via hooks to properly manage
        # types.

    def forward(self, input: Tensor) -> Tensor:
        return linear_nf4(input=input, weight=self.weight)
