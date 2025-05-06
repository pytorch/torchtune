
import torch

class DeepSeekV3MoE(nn.Module):
    def __init__(self):
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, h = x.shape
        top_scores, token_idxs, num_tokens_per_expert = self.router(x.reshape(b * s, h))
        token_idxs = token_idxs.reshape(-1, 1).expand(-1, h)
        routed_input = torch.gather(x.view(-1, h), dim=0, index=token_idxs)
        routed_input = routed_input.reshape(b, s, h)
        return self.experts(routed_input)
    
class DeepSeekV3TokenChoiceTopKRouter(nn.Module):
    def __init__(self):
        self.gate = gate # nn.Linear
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.n_groups = n_groups
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.experts)))
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        scores = self.gate(x)
        scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)
        
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(n, self.n_groups, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                n, self.n_groups, self.n_routed_experts // self.n_groups
            )
            .reshape(n, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        if self.num_experts > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = (
            topk_weight * self.routed_scaling_factor
        )  # must multiply the scaling factor

        return topk_idx, topk_weight