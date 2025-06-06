# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch
from torch import nn
from typing import Optional


class DeepseekV3MoE(nn.Module):
    """This class implements the Mixture of Experts (MoE) layer for DeepSeek V3.
    This comprises a set of a router and a set of experts, which are typically smaller than MLP layers in standard
    transformer models. The router is used to select a subset of experts for each token, and the selected experts are
    then used to compute the output of the MoE layer. See more details in https://arxiv.org/2401.0606.

    Args:
        experts (nn.Module): experts module.
        router (nn.Module): router module.
        shared_expert (Optional[nn.Module]): shared expert module. Default is None.
    """

    def __init__(
        self,
        *,
        experts: nn.ModuleDict,
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

        b, s, dim = x.shape
        # top_scores and selected_indices shape (bs*slen*experts_per_token,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
        ) = self.router(x.reshape(b * s, dim))

        token_indices = token_indices.unsqueeze(1).expand(-1, dim)
        # shape (b*s*experts_per_token, dim)

        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )

        # shape (b*s*top_k, dim)
        routed_input = torch.split(routed_input, split_size_or_sections=num_tokens_per_expert.tolist(), dim=0)
        routed_output = []
        for expert_idx, x_expert in enumerate(routed_input):
            if x_expert.numel() == 0:
                routed_output.append(torch.zeros_like(x_expert))
                continue
            routed_output.append(self.experts[str(expert_idx)](x_expert))

        routed_output = torch.cat(routed_output, dim=0)
        routed_output = routed_output * top_scores.unsqueeze(-1)

        out = torch.zeros_like(x.reshape(b * s, dim)).to(routed_output.dtype)
        if routed_output.numel() > 0:
            out.scatter_add_(dim=0, index=token_indices, src=routed_output)

        out = out.view(b, s, dim).to(x.dtype)

        if self.shared_expert is not None:
            out = out + self.shared_expert(x)

        return out


class DeepSeekV3TokenChoiceTopKRouter(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: int,
                 experts_per_token: int,
                 num_groups: int,
                 topk_groups: int,
                 norm_topk_prob: bool,
                 routed_scaling_factor: float
                 ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.num_groups = num_groups
        self.topk_groups = topk_groups
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = nn.Parameter(torch.rand((self.num_experts)))
        self.gate = nn.Parameter(torch.empty((num_experts, dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        logits = F.linear(x.to(torch.float32), self.gate.to(torch.float32), None)

        # calculate scores for every expert in every group
        scores = torch.sigmoid(logits)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # now calculate scores for every group based on the
        # top 2 scores of experts within each group
        experts_per_group = self.num_experts // self.num_groups
        group_scores = (
            scores_for_choice.view(n, self.num_groups, experts_per_group)
            .topk(2, dim=-1)[0].sum(dim=-1)
        )

        # grab the topk_groups number of groups based
        # on the scores for each group calculated above
        group_idxs = torch.topk(group_scores, k=self.topk_groups, dim=-1, sorted=False).indices

        # mask out all experts within groups which will not be considered
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idxs, True)  # [n, n_group]

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                n, self.num_groups, experts_per_group
            )
            .reshape(n, -1)
        )
        # masked_scores = scores + self.e_score_correction_bias.unsqueeze(0)
        masked_scores = scores_for_choice.masked_fill(
            ~score_mask, float('-inf')
        )
        # now select the top experts_per_token number of
        # experts based on experts within eligible groups
        _, selected_experts_idxs = torch.topk(masked_scores, k=self.experts_per_token, dim=-1, sorted=False)
        scores_per_token = scores.gather(1, selected_experts_idxs)

        # normalize scores
        if self.num_experts > 1 and self.norm_topk_prob:
            denominator = scores_per_token.sum(dim=-1, keepdim=True) + 1e-20
            scores_per_token /= denominator

        # apply scaling factor
        scores_per_token = scores_per_token * self.routed_scaling_factor

        num_tokens_per_expert = torch.histc(
            selected_experts_idxs.float(), bins=self.num_experts, min=0, max=self.num_experts - 1
        ).to(torch.int32)

        token_idxs_experts_sorted = torch.argsort(
            selected_experts_idxs.view(-1), stable=True
        )

        scores_per_expert = scores_per_token.view(-1)[token_idxs_experts_sorted]
        token_idxs_experts_sorted = (
            token_idxs_experts_sorted // self.experts_per_token
        )
        return scores_per_expert, token_idxs_experts_sorted, num_tokens_per_expert
