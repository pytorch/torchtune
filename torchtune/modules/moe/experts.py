# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, List

import torch
from torch import nn
from torch.nn import functional as F
from torchtune.modules.peft import AdapterModule


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        activation (Callable): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        num_experts: int = 1,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.act_fn = activation

    def reset_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        if self.up_proj is not None:
            nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

    # TODO: force no inference mode as a hack to get around
    # "Cannot set version_counter for inference tensor"
    @torch.inference_mode(mode=False)
    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``
                enumerating the number of tokens each expert receives

        Returns:
            torch.Tensor: tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
        """

        # a tuple of tensors indexed by experts
        # each with shape (tokens_per_expert(varying), dim)
        x = torch.split(
            x,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        out_experts_splits = []
        for expert_idx, x_expert in enumerate(x):
            w1, w2, w3 = (
                self.gate_proj[expert_idx],
                self.down_proj[expert_idx],
                self.up_proj[expert_idx],
            )
            h = self.act_fn(torch.matmul(x_expert, w1))
            h = h * torch.matmul(x_expert, w3)
            h = torch.matmul(h, w2)
            # h shape (tokens_per_expert(varying), dim)
            out_experts_splits.append(h)
        out = torch.cat(out_experts_splits, dim=0)

        return out


class LoRAGroupedExperts(nn.Module, AdapterModule):
    """This class implements the grouped experts layer used in Mixture of Experts with additional LoRA
    adapter parameters.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        rank (int): rank of the low-rank approximation
        alpha (float): scaling factor for the low-rank approximation
        dropout (float): dropout probability before LoRA layer. Default: 0.0
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        activation (Callable): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        num_experts: int = 1,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.rank = rank
        self.alpha = alpha
        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.act_fn = activation

        # 'self.disabled' is a flag showing whether to turn off LoRA adapters,
        # this can be used in DPO for treating the lora adapters as the policy model
        # and disabling it to treat the base model as the reference model
        self.disabled = False
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.lora_gate_a = nn.Parameter(torch.empty(num_experts, dim, rank))
        self.lora_gate_b = nn.Parameter(torch.empty(num_experts, rank, hidden_dim))
        self.lora_down_a = nn.Parameter(torch.empty(num_experts, hidden_dim, rank))
        self.lora_down_b = nn.Parameter(torch.empty(num_experts, rank, dim))
        self.lora_up_a = nn.Parameter(torch.empty(num_experts, dim, rank))
        self.lora_up_b = nn.Parameter(torch.empty(num_experts, rank, hidden_dim))
        self.merged = False
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        if self.up_proj is not None:
            nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

        nn.init.kaiming_uniform_(self.lora_gate_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_gate_b)
        nn.init.kaiming_uniform_(self.lora_down_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_down_b)
        if self.lora_up_a is not None:
            nn.init.kaiming_uniform_(self.lora_up_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up_b)

    def adapter_params(self) -> List[str]:
        """
        Return a list of strings corresponding to the names of the ``nn.Parameter`` s in
        the model coming from the adapter.

        For LoRA this means lora_gate, lora_up, lora_down a and b weights.
        """
        # NOTE: this function has to be updated if the names of the lora parameters
        # in this module change.
        adapter_params = [
            "lora_gate_a",
            "lora_gate_b",
            "lora_down_a",
            "lora_down_b",
            "lora_up_a",
            "lora_up_b",
        ]
        return adapter_params

    def _lora_tc_layer_forward(
        self,
        x: torch.Tensor,
        base_weight: torch.Tensor,
        lora_a_weight: torch.Tensor,
        lora_b_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass a single linear layer with lora adapter layers for Token Choice routing.

        Args:
            x (torch.Tensor): Input tensor with shape ``(tokens_per_expert, in_dim)``.
            base_weight (torch.Tensor): weight of the base linear projection, shape ``(in_dim, out_dim)``.
            lora_a_weight (torch.Tensor): weight of the lora adapter A layer, shape ``(in_dim, rank)``.
            lora_b_weight (torch.Tensor): weight of the lora adapter B layer, shape ``(rank, out_dim)``.

        Returns:
            torch.Tensor: Output tensor with shape ``(tokens_per_expert, out_dim)``.
        """
        out = torch.matmul(x, base_weight)
        if self.disabled:
            return out
        lora_out = torch.matmul(self.dropout(x), lora_a_weight)
        lora_out = (self.alpha / self.rank) * torch.matmul(lora_out, lora_b_weight)
        return out + lora_out

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``
                enumerating the number of tokens each expert receives

        Returns:
            torch.Tensor: Tuple of input tensors each with shape ``(num_experts, tokens_per_expert, dim)`` for Token Choice(TC)
                or a single tensor with shape (num_experts, tokens_per_expert, dim) for Expert Choice(EC).
        """
        # a tuple of tensors indexed by experts
        # each with shape (tokens_per_expert(varying), dim)
        x = torch.split(
            x,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        out_experts_splits = []
        for expert_idx, x_expert in enumerate(x):
            gate_proj, down_proj = (
                self.gate_proj[expert_idx],
                self.down_proj[expert_idx],
            )
            lora_gate_a, lora_gate_b, lora_down_a, lora_down_b = (
                self.lora_gate_a[expert_idx],
                self.lora_gate_b[expert_idx],
                self.lora_down_a[expert_idx],
                self.lora_down_b[expert_idx],
            )
            h = self.act_fn(
                self._lora_tc_layer_forward(
                    x_expert, gate_proj, lora_gate_a, lora_gate_b
                )
            )

            if self.up_proj is not None:
                up_proj = self.up_proj[expert_idx]
                lora_up_a, lora_up_b = (
                    self.lora_up_a[expert_idx],
                    self.lora_up_b[expert_idx],
                )
                h = h * self._lora_tc_layer_forward(
                    x_expert, up_proj, lora_up_a, lora_up_b
                )

            h = self._lora_tc_layer_forward(h, down_proj, lora_down_a, lora_down_b)

            # h shape (tokens_per_expert(varying), hidden_dim)
            out_experts_splits.append(h)
        out = torch.cat(out_experts_splits, dim=0)

        return out
