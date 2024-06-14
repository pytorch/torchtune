# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch


def get_rewards(
    scores: torch.Tensor,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    kl_controller_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the rewards for the given scores, logprobs, and reference logprobs.

    Args:
        scores (torch.Tensor): Reward model scores, shape (b,).
        logprobs (torch.Tensor): Policy logprobs, shape (b, reponse_len).
        ref_logprobs (torch.Tensor): Reference base model, shape (b, reponse_len).
        kl_controller_value (float): Adaptive KL controller value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple of three tensors with shape [b, response_len] each:
            - total_reward
            - kl between policy and reference base model
            - reward corresponding to kl above

    Notation used for tensor shapes:
        - b: batch size
        - response_len: model response length
    """

    # 1. calculate kl between logprobs and reflogprobs
    # 2. calculate kl reward using adaptive scaling value
    # 3. calculate total reward by summing above
    # return all
    kl = logprobs - ref_logprobs
    kl_reward = -kl_controller_value * kl

    total_reward = kl_reward.clone()

    # adding reward to kl at final position
    # https://github.com/openai/lm-human-preferences/blob/cbfd210bb8b08f6bc5c26878c10984b90f516c66/lm_human_preferences/train_policy.py#L153
    total_reward[:, -1] += scores

    return total_reward, kl, kl_reward


def whiten(x: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """
    Whitens (normalizes) the input tensor.

    Args:
        advantages (torch.Tensor): The advantages.

    Returns:
        torch.Tensor: The whitened tensor.
    """
    mean, var = x.mean(), x.var()
    whitened = (x - mean) * torch.rsqrt(var + 1e-8)
    if shift_mean:
        whitened += mean
    return whitened


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values. Taken from
    https://github.com/huggingface/trl/blob/main/trl/core.py"""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(
    x: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values. Taken from
    https://github.com/huggingface/trl/blob/main/trl/core.py"""
    mean = masked_mean(x, mask)
    centered_values = x - mean
    var = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        var = var * bessel_correction
    return var


def masked_whiten(
    x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values. Taken from
    https://github.com/huggingface/trl/blob/main/trl/core.py"""
    mean = masked_mean(x, mask)
    var = masked_var(x, mask) if mask.any else x.var()
    whitened = (x - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def estimate_advantages(
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lmbda: float,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the advantages and returns for the PPO algorithm using Generalized Advantage Estimation
    https://arxiv.org/pdf/1506.02438.pdf.

    Args:
        values (torch.Tensor): The predicted values for each state. Shape: (b, reponse_len)
        rewards (torch.Tensor): The rewards received at each time step. Shape: (b, reponse_len)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the estimated advantages and returns.
            - advantages (torch.Tensor): The estimated advantages. Shape: (b, reponse_len)
            - returns (torch.Tensor): The estimated returns. Shape: (b, reponse_len)
            - masks (torch.Tensor): A boolean tensor for excluding invalid rewards/values. Shape: (b, reponse_len)
    Notation:
        - b: batch size
        - reponse_len: model response length
    """

    last_gae_lam = 0
    advantages_reversed = []

    reponse_length = values.shape[-1]

    # estimate advantage for every predicted token position
    for t in reversed(range(reponse_length)):
        # value of the next state
        next_values = values[:, t + 1] if t < reponse_length - 1 else 0.0
        # exponentially discounted temporal difference error:
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        # GAE-Lambda advantage discouting saved for the next iteration
        # as A_t = delta_t + gamma * lambda * A_{t+1} + ...
        last_gae_lam = delta + gamma * lmbda * last_gae_lam
        advantages_reversed.append(last_gae_lam)

    advantages = torch.stack(advantages_reversed[::-1], axis=1)

    # returns are the expected value of taking action a_t at each timepoint over
    # a trajectory. the value estimates v_t are the expected value over all actions
    # over a trajectory - the advantage is the difference between the two
    returns = advantages + values

    # normalize advantages across the batch of trajectories to reduce variance
    if masks is not None:
        advantages = masked_whiten(advantages, masks)
    else:
        advantages = whiten(advantages)

    return advantages, returns
