# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
from typing import Tuple, Dict, Any, Optional, Union, List

import torch.nn.functional as F

from torch import nn, Tensor

from torchao.dtypes.nf4tensor import linear_nf4, NF4Tensor
from torchtune.modules.low_precision import (  # noqa: F401
    _register_nf4_dispatch_ops,
    FrozenNF4Linear,
)
from torchao.dtypes.nf4tensor import SubclassTensorArgs
from torchtune.modules.peft.peft_utils import AdapterModule


class LoRALinear(nn.Module, AdapterModule):
    """LoRA linear layer as introduced in `LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

    LoRA perturbs a given layer via a low-rank approximation where only
    the rank decomposition matrices are trainable. In a linear layer instead of
    :math:`x \\mapsto W_0x` a LoRALinear layer is defined as
    :math:`x \\mapsto W_0x + (\\alpha / r)BAx`, where :math:`r` is the rank of
    the matrices :math:`A` and :math:`B` and :math:`\\alpha` is a scaling factor.
    As in the original implementation, we support dropout before multiplication
    by the low-rank matrices.

    Args:
        in_dim (int): input dimension
        out_dim (int): output dimension
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability. Default: 0.0
        use_bias (bool): whether to include bias in the original linear layer.
            Default: False
        quantize_base (bool): Whether to quantize base linear weight or not.
            Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        self.use_bias = use_bias
        self._quantize_base = quantize_base
        weight, bias = self._create_weight_and_bias()
        self.register_parameter("weight", nn.Parameter(weight))
        self.register_parameter(
            "bias", nn.Parameter(bias) if bias is not None else None
        )
        self.dropout = nn.Dropout(p=dropout)
        self.lora_a = nn.Linear(in_features=in_dim, out_features=rank, bias=False)
        self.lora_b = nn.Linear(in_features=rank, out_features=out_dim, bias=False)
        self.merged = False
        # Note: FSDP's meta device initialization contract assumes that a module's
        # reset_parameters method only initializes its own parameters (i.e. no child
        # params are initialized, as is done in initialize_parameters below).
        # For that reason, we patch reset_parameters directly on lora_a and lora_b submodules
        # when using meta device. This is done in
        # torchtune.utils.prepare_model_for_fsdp_with_meta_device.
        # See this issue for more details: https://github.com/pytorch/pytorch/issues/104187.
        # Without meta device, we only need the following:
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        _lora_a_init_params(self.lora_a)
        _lora_b_init_params(self.lora_b)

    def _create_weight_and_bias(self):
        """
        Creates a linear weight and bias tensor, using NF4 dtype if we're quantizing
        (indicated via quantize_base=True).
        """
        in_dim, out_dim, use_bias = self.in_dim, self.out_dim, self.use_bias
        linear = (
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
            if not self._quantize_base
            else FrozenNF4Linear(in_dim, out_dim, bias=False)
        )
        weight = linear.weight
        bias = None
        if self.use_bias:
            if self._quantize_base:
                raise NotImplementedError(
                    "Quantized LoRALinear does not support bias at the moment."
                )
            bias = linear.bias
        return weight, bias

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        # NOTE: this function has to be updated if the names of "lora_a" and "lora_b"
        # in this module change.
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        return adapter_params

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        """
        if self._quantize_base:
            out = linear_nf4(input=x, weight=self.weight)
        else:
            out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out

    def fsdp_extensions(self) -> Dict[str, Any]:
        if isinstance(self.weight._local_tensor, NF4Tensor):
            from torch.distributed._composable.fsdp import FSDPTensorExtensions
            weight_extensions = FSDPTensorExtensions(
                self._fsdp_pre_all_gather, self._fsdp_post_all_gather
            )
            return {"weight": weight_extensions}
        else:
            return {}

    def _fsdp_pre_all_gather(self, sharded_param: torch.Tensor):
        return (
            sharded_param.quantized_scalers,
            sharded_param.quantization_factor,
            sharded_param.quantized_data,
        ), (
            SubclassTensorArgs(
                sharded_param.size(),
                sharded_param.stride(),
                sharded_param.storage_offset(),
                sharded_param.dtype,
                sharded_param.device,
                sharded_param.requires_grad,
            ),
            sharded_param.block_size,
            sharded_param.n_blocks,
            sharded_param.scaler_block_size,
            sharded_param.scaler_mean,
            sharded_param.nf4,
        )

    def _fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[NF4Tensor, Tuple[torch.Tensor, ...]], None]:
        (quantized_scalers, quantization_factor, quantized_data) = all_gather_outputs
        (tensor_meta, block_size, n_blocks, scaler_block_size, scaler_mean, nf4)  = metadata
        tensor_meta.original_shape = torch.Size([quantized_data.size(0) * 2])
        if out is not None:
            assert isinstance(out, NF4Tensor), f"{type(out)}"
            assert (
                quantized_scalers.untyped_storage().data_ptr()
                == out.quantized_scalers.untyped_storage().data_ptr() and
                quantization_factor.untyped_storage().data_ptr()
                == out.quantization_factor.untyped_storage().data_ptr() and
                quantized_data.untyped_storage().data_ptr()
                == out.quantized_data.untyped_storage().data_ptr()
            ), f"Expects out's data to be the all-gather output"
            return

        return NF4Tensor(
            tensor_meta,
            block_size,
            n_blocks,
            scaler_block_size,
            quantized_scalers,
            quantization_factor,
            scaler_mean,
            quantized_data,
            nf4,
        ), (quantized_scalers, quantization_factor, quantized_data)


def _lora_a_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA A weight to Kaiming uniform.
    """
    nn.init.kaiming_uniform_(x.weight, a=math.sqrt(5))


def _lora_b_init_params(x: nn.Linear) -> None:
    """
    Initialize LoRA B weight to zeros.
    """
    nn.init.zeros_(x.weight)
