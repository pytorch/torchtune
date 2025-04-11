# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn


class TokenChoiceTopKRouter(nn.Module):
    """This class implements Token Choice routing. In Token Choice top K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        experts_per_token: int,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid is performed in float32 to avoid loss explosion
        scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)

        # top scores shape (bs*slen, top_k)
        top_scores, selected_experts_indices = torch.topk(
            scores, k=self.experts_per_token, dim=1
        )
        self.selected_experts_indices = selected_experts_indices
        # top_scores /= top_scores.sum(dim=-1, keep_dim=True).to(x.dtype)

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.experts_per_token
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert


class MoE(nn.Module):
    """This class implements the moe layer which is Mixture of Experts. Mixture of Experts
    typically consists of a set of expert networks, alongside with a router, which directs input tokens
    to the appropriate experts. See more details in https://arxiv.org/pdf/2407.06204.

    Args:
        experts (nn.Module): experts module.
        router (nn.Module): router module.
        shared_expert (Optional[nn.Module]): shared expert module. Default is None.
    """

    def __init__(
        self,
        *,
        experts: nn.Module,
        router: nn.Module,
        shared_expert: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        # top_scores and selected_indices shape (bs*slen*experts_per_token,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim))

        # shape (bs*slen*experts_per_token, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*experts_per_token, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )
        routed_input = routed_input * top_scores.reshape(-1, 1)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x).reshape(bs * slen, dim)
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))
        out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out
