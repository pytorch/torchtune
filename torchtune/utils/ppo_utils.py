# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtune.utils._generation import sample, update_stop_tokens_tracker


def left_padded_collate(
    batch: List[Dict[str, List[int]]], max_seq_len: int, padding_idx: int = 0
) -> torch.Tensor:
    """
    Pads a batch of sequences with left padding to the maximum sequence length in the batch.
    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing inputs.
        max_seq_len (int): The maximum sequence length to pad to.
        padding_idx (int): The padding index. Defaults to 0.
    Returns:
        torch.Tensor: The padded tensor of input ids with shape [batch_size, max_seq_len].

    """
    pad_toks = pad_sequence(
        [torch.tensor(x["tokens"][::-1]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    seq_idxs_rev = torch.arange(max_seq_len - 1, -1, -1)
    return torch.stack([tok[seq_idxs_rev] for tok in pad_toks])


class AdaptiveKLController:
    """
    A class that implements an adaptive KL controller from https://arxiv.org/pdf/1909.08593.pdf.

    Attributes:
        value (float): The initial KL coefficient value.
        target (float): The target KL value.
        horizon (int): The horizon for the update calculation.

    Methods:
        update(current, n_steps): Updates the KL coefficient value based on the current KL value and the number of steps.
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: float):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int) -> None:
        """
        Updates the KL coefficient value based on the current KL value and the number of steps.

        Args:
            current (float): The current KL value.
            n_steps (int): The number of steps.
        """
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


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
    mean, var = x.mean(), x.var(unbiased=False)
    whitened = (x - mean) * torch.rsqrt(var + 1e-8)
    if shift_mean:
        whitened += mean
    return whitened


def estimate_advantages(
    values: torch.Tensor, rewards: torch.Tensor, gamma: float, lmbda: float
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
    advantages = whiten(advantages)

    return advantages, returns


def generate_next_token(
    model: torch.nn.Module,  # TODO (SalmanMohammadi) leaving as nn.module until TransformerDecoder refactor
    input_pos: torch.Tensor,
    x: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos, mask=mask)[:, -1]
    return sample(logits, temperature, top_k)


def generate_next_token_with_value_head_model(
    model: torch.nn.Module,  # TODO (SalmanMohammadi) leaving as nn.module until TransformerDecoder refactor
    input_pos: torch.Tensor,
    x: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits, _ = model(x, input_pos=input_pos, mask=mask)
    return sample(logits[:, -1], temperature, top_k)


def get_causal_mask(
    tokens: torch.Tensor,
    *,
    padding_id: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generates a causal attention mask for the given tokens with padding values
    correctly masked out, suitable for consumtion by ~torch.nn.functional.scaled_dot_product_attention~
    where the mask is added to the attention score.

    Args:
        tokens (torch.Tensor): tensor of token IDs with shape [bsz x seq_length]
        padding_id (int): token ID to use for padding, default 0.
        dtype (torch.dtype): dtype to infer fill value for masking
    Returns:
        torch.Tensor: Casual mask with shape [bsz x seq_length x seq_length]
    """
    fill_value = torch.finfo(dtype).min
    mask = torch.triu(
        torch.full((tokens.shape[-1], tokens.shape[-1]), fill_value), diagonal=1
    ).to(tokens.device, dtype=dtype)
    padding_mask = (
        (tokens == padding_id).unsqueeze(1).expand(-1, tokens.shape[1], tokens.shape[1])
    )
    mask = mask.masked_fill(padding_mask, fill_value)
    return mask


@torch.inference_mode()
def generate(
    model,
    prompt,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k=None,
    stop_tokens=None,
    custom_generate_next_token=None,
    dtype: torch.dtype = torch.float32,
):
    """
    Generates tokens from a model conditioned on a prompt. In contrast to ~torchtune.utils._generation.generate~,
    this function handles input sequences which may be left-padded by shifting position IDs
    and correctly masking out padding tokens.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape either [seq_length] or [bsz x seq_length]. Padded sequences should be
            left-padded with `pad_id` for sequence collation.
        max_generated_tokens (int): number of tokens to be generated
        pad_id (int): token ID to use for padding, default 0.
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities,
            default None.
        stop_tokens (Optional[List[int]]): If specified, generation is stopped when any of these tokens are generated,
            default None.
        custom_generate_next_token (Optional[Callable]): If specified, we'll use the ``custom_generate_next_token function``.
            This is generally only useful if you want to specify a ``torch.compile`` version of the generate next token for
            performance reasons. If None, we use the default ``generate_next_token`` function. Default is None.

    Examples:
        >>> model = torchtune.models.llama3.llama3_8b()
        >>> tokenizer = torchtune.models.llama3.llama3_tokenizer()
        >>> prompt = [0, 0, 0] + tokenizer("Hi my name is") # substitute 0 with pad_id
        >>> output = generate(model, torch.tensor(prompt), max_generated_tokens=100)
        >>> print(tokenizer.decode(output[0]))
        ?? ?? ?? Hi my name is Jeremy and I'm a friendly language model assistant!

    Returns:
        List[List[int]]: collection of lists of generated tokens
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()
    generated_tokens[generated_tokens == pad_id] = 0

    if stop_tokens is not None:
        # convert stop tokens to tensor for easy matching
        stop_tokens = (
            torch.tensor(stop_tokens, device=prompt.device) if stop_tokens else None
        )
        # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
        stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
        # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
        # that already hit a stop token
        stop_token_mask = torch.ones(
            (bsz, prompt_length), dtype=torch.int32, device=prompt.device
        )

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    for i in range(max_generated_tokens):
        # update stop_token_mask if we reached a stop token in a previous step
        # by appending the logical not of stop_token_reached to the end of the mask
        # reshaped to be bsz first
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        padding_mask = generated_tokens == pad_id
        mask = get_causal_mask(generated_tokens, padding_id=pad_id, dtype=dtype)
        if padding_mask is not None:
            input_pos = (~padding_mask).cumsum(-1) - (~padding_mask).long()
            input_pos = input_pos.to(device=generated_tokens.device, dtype=torch.int)
        else:
            input_pos = torch.arange(
                0, prompt_length + i, device=generated_tokens.device
            )

        tokens = custom_generate_next_token(
            model,
            input_pos=input_pos,
            x=generated_tokens,
            mask=mask,
            temperature=temperature,
            top_k=top_k,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if stop_token_reached.all().item():
                break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens = generated_tokens * stop_token_mask
        # if pad_id is not 0, replace 0 with pad_id
        if pad_id != 0:
            generated_tokens[generated_tokens == 0] = pad_id

    return generated_tokens
