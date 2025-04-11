# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


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
            torch.Tensor: tensor with shape (bsz * seq_len * experts_per_token, dim)
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
