from typing import Dict, List

import torch
from tensordict import TensorClass


class Trajectory(TensorClass["nocast"]):
    query_responses: torch.Tensor
    responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    query_response_padding_masks: torch.Tensor
    seq_lens: torch.Tensor
    answers: torch.Tensor
    policy_version: int
    rewards: torch.Tensor
    advantages: torch.Tensor
    successes: torch.Tensor
    reward_metadata: Dict[str, List[str]]
