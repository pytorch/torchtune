# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional

import torch
from torchtune.modules import TransformerDecoder


def multinomial_sample_one(probs):
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(logits, temperature=1.0, top_k=None):
    """Samples from a probability distribution."""
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_sample_one(probs)


def generate_next_token(model, input_pos, x, temperature=1.0, top_k=None):
    """Generates the next token."""
    logits = model(x, input_pos)[:, -1]
    return sample(logits, temperature, top_k)


@torch.inference_mode()
def generate(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[List[int]] = None,
    custom_generate_next_token: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Generates tokens from a model conditioned on a prompt.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt
        max_generated_tokens (int): number of tokens to be generated
        temperature (float): value to scale the predicted logits by
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities
        stop_tokens (Optional[List[int]]): If specified, generation is stopped when any of these tokens are generated
        custom_generate_next_token (Optional[Callable]): If specified, we'll use the custom generate_next_token function

    Returns:
        List: list of generated tokens
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    stop_tokens = (
        torch.tensor(stop_tokens, device=prompt.device) if stop_tokens else None
    )
    bsz, seq_length = prompt.size()
    generated_tokens = prompt.clone()
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    stop_token_mask = torch.ones(
        (bsz, seq_length + 1), dtype=torch.int32, device=prompt.device
    )

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    # generate the first token conditioned on the prompt
    token = generate_next_token(
        model,
        input_pos=torch.arange(0, seq_length, device=prompt.device),
        x=prompt,
        temperature=temperature,
        top_k=top_k,
    )
    generated_tokens = torch.cat([generated_tokens, token], dim=-1)

    # stop early if we reach a stop token in every seq
    if stop_tokens is not None:
        stop_token_reached_curr = torch.isin(token, stop_tokens)
        stop_token_reached |= stop_token_reached_curr.flatten()
        if stop_token_reached.all().item():
            return generated_tokens.tolist()

    input_pos = torch.tensor([seq_length], device=prompt.device)
    for _ in range(max_generated_tokens - 1):
        # update stop_token_mask if we reached a stop token in a previous step
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        token = custom_generate_next_token(
            model, input_pos=input_pos, x=token, temperature=temperature, top_k=top_k
        )

        generated_tokens = torch.cat([generated_tokens, token], dim=-1)
        input_pos += 1

        if stop_tokens is not None:
            stop_token_reached_curr = torch.isin(token, stop_tokens)
            stop_token_reached |= stop_token_reached_curr.flatten()
            if stop_token_reached.all().item():
                break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens = generated_tokens * stop_token_mask

    return generated_tokens.tolist()
