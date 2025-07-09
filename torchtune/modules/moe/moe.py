# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torch import nn

from .utils import should_use_grouped_mm


class TokenChoiceTopKRouter(nn.Module):
    """This class implements Token Choice routing. In Token Choice top K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
        norm_topk_prob (bool): Whether to normalize the topk probabilities.
        softmax (bool): use softmax if true and sigmoid if false
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        experts_per_token: int,
        norm_topk_prob: bool = False,
        softmax: bool = False,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.norm_topk_prob = norm_topk_prob
        self.softmax = softmax

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
        if self.softmax:
            scores = nn.functional.softmax(scores, dim=1, dtype=torch.float32).to(
                x.dtype
            )
        else:
            scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)

        # top scores shape (bs*slen, top_k)
        top_scores, selected_experts_indices = torch.topk(
            scores, k=self.experts_per_token, dim=1
        )
        self.selected_experts_indices = selected_experts_indices
        if self.norm_topk_prob:
            top_scores /= top_scores.sum(dim=-1, keepdim=True).to(x.dtype)

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
        scale_after_fwd (bool): if True, scale routed outputs by router scores instead of inputs.
    """

    def __init__(
        self,
        *,
        experts: nn.Module,
        router: nn.Module,
        shared_expert: Optional[nn.Module] = None,
        scale_after_fwd: bool = False,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert
        self.use_grouped_mm = should_use_grouped_mm()
        self.scale_after_fwd = scale_after_fwd

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
        if not self.scale_after_fwd:
            routed_input = routed_input * top_scores.reshape(-1, 1)

        if self.use_grouped_mm:
            # NOTE: In order to use torch._grouped_mm, we need to make sure
            # the number of tokens each expert gets is a multiple of 16.
            # The following kernel helps achieve this via padding, without
            # incurring synchronization between device and host.
            from torchtune.modules.moe.indices import generate_permute_indices

            ALIGN_SIZE_M = 16  # noqa

            with torch.no_grad():
                (
                    permuted_indices,
                    num_tokens_per_expert,
                    _,
                ) = generate_permute_indices(
                    num_tokens_per_expert,
                    self.experts.num_experts,
                    1,
                    ALIGN_SIZE_M,
                )
            token_indices = torch.vstack(
                (token_indices, token_indices.new_zeros((dim)))
            )
            token_indices = token_indices[permuted_indices, :]
            routed_input = torch.vstack((routed_input, routed_input.new_zeros((dim))))
            routed_input = routed_input[permuted_indices, :]
            if self.scale_after_fwd:
                top_scores = torch.cat((top_scores, top_scores.new_zeros(1)), dim=0)
                top_scores = top_scores[permuted_indices]

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        if self.scale_after_fwd:
            routed_output = routed_output * top_scores.reshape(-1, 1)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x).reshape(bs * slen, dim)
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))

        if self.use_grouped_mm:
            num_tokens = num_tokens_per_expert.sum().item()
            if torch.compiler.is_compiling():
                # Hints to compile dynamic shapes to pass through slice shape checks.
                torch._check_is_size(num_tokens)
                torch._check(num_tokens <= token_indices.size(0))
                torch._check(num_tokens <= routed_output.size(0))
            out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        else:
            out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out
