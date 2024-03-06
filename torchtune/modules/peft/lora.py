# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from typing import List, Optional

import torch

import torch.nn.functional as F

from torch import nn, Tensor

from torchtune.modules.peft.peft_utils import AdapterModule
from torchtune.utils.tensor_utils import _copy_tensor


def reset_lora_params(model: nn.Module, device: torch.device) -> None:
    """
    Initializes lora parameters of a given model. This is useful
    if model is initialized on meta device and custom initialization
    needs to be run for LoRA parameters. This method is meant to be used
    in tandem with ``LoRALinear``'s ``reset_lora_parameters`` and simply
    calls this method on each instance.

    Args:
        model (nn.Module): Instance of model class containing LoRA parameters
        device (torch.device): Device to initialize LoRA parameters on.
    """
    for m in model.modules():
        if hasattr(m, "reset_lora_parameters"):
            m.reset_lora_parameters(device=device)


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
        use_bias_in_lora_matrices (bool): whether to add biases to the LoRA matrices
            A and B. Default: False
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        use_bias_in_lora_matrices: bool = False,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.out_dim = out_dim
        linear = nn.Linear(in_features=in_dim, out_features=out_dim, bias=use_bias)
        # Clone weight / bias directly into the LoRALinear, for 1:1 mapping with how Linear layers are used in
        # vanilla Transformers.
        self.register_parameter("weight", nn.Parameter(_copy_tensor(linear.weight)))
        if use_bias:
            self.register_parameter("bias", nn.Parameter(_copy_tensor(linear.bias)))
        else:
            self.register_parameter("bias", None)
        self.dropout = nn.Dropout(p=dropout)
        self.use_bias_in_lora_matrices = use_bias_in_lora_matrices
        self.lora_a = nn.Linear(
            in_features=in_dim, out_features=rank, bias=self.use_bias_in_lora_matrices
        )
        self.lora_b = nn.Linear(
            in_features=rank, out_features=out_dim, bias=self.use_bias_in_lora_matrices
        )
        self._lora_params_initialized = False
        # Skip init if we are under a meta device context
        if not self.weight.is_meta:
            self.reset_lora_parameters()

    def reset_lora_parameters(self, device: Optional[torch.device] = None):
        # Initialize as in
        # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L119
        # TODO: getting default / current device with torch.empty(1).device - replace with torch.get_default_device
        # once available in latest stable version.
        init_device = device if device is not None else torch.empty(1).device
        # Should not be initializing on a meta device
        assert init_device != torch.device("meta")
        self.lora_a.to_empty(device=init_device)
        self.lora_b.to_empty(device=init_device)
        nn.init.zeros_(self.lora_b.weight)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        self._lora_params_initialized = True

    def adapter_params(self) -> List[str]:
        """
        Return lora_a.weight and lora_b.weight as adapter params.
        If bias is enabled, also return lora_a.bias and lora_b.bias.
        """
        adapter_params = ["lora_a.weight", "lora_b.weight"]
        if self.use_bias_in_lora_matrices:
            adapter_params.extend(["lora_a.bias", "lora_b.bias"])
        return adapter_params

    # @functools.cached_property
    def _lora_delta(self):
        return (self.alpha / self.rank) * (self.lora_b.weight @ self.lora_a.weight)

    @torch.no_grad
    def merge_lora_weights(self, *args, **kwargs):
        self.weight += self._lora_delta
        self.cached_lora_a_weight = self.lora_a.weight
        self.cached_lora_b_weight = torch.clone(self.lora_b.weight)
        # same for bias
        # is there a better way to unregister module?
        del self.lora_a
        del self.lora_b

    @torch.no_grad
    # This has to run after calling state_dict on self and children
    def unmerge_lora_weights(self, *args, **kwargs):
        self.lora_a = nn.Linear()
        # self.cached_lora_a
        self.lora_b = self.cached_lora_b
        del self.cached_lora_a
        del self.cached_lora_b
        self.weight -= self._lora_delta

    # def register_merging_state_dict_hooks(self):
    #     self._register_pre_state_dict_hook(self.pre_state_dict_hook)
    #     self._register_state_dict_hook(self.post_state_dict_hook)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape ``(..., in_dim)``

        Returns:
            Tensor: output tensor with shape ``(..., out_dim)``

        Raises:
            RuntimeError: if reset_lora_params was never called
        """
        if not self._lora_params_initialized:
            raise RuntimeError(
                "lora reset_lora_params was never called, please file a bug."
            )
        out = F.linear(x, self.weight, self.bias)
        lora_out = self.lora_a(self.dropout(x))
        lora_out = (self.alpha / self.rank) * self.lora_b(lora_out)
        return out + lora_out
