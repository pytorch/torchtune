# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Set

import torch

from torchtune.modules import TransformerDecoder


def multinomial_sample_one(probs):
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None
) -> torch.Tensor:
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)

    # keep only the top_k logits if this is specified
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    # compute the probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # sample the next token
    token = multinomial_sample_one(probs)
    return token


def generate_next_token(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    # x: [1, s]
    # input_pos: [s]
    logits = model(x, input_pos)

    # logits: [1, s, v] where v is vocab_size
    # for sampling we extract the logits for the
    # last token and convert to shape: [v]
    logits = logits[0, -1]

    # sample the next token
    token = sample(logits, temperature, top_k)
    return token


@torch.inference_mode()
def generate(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    max_generated_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[Set[int]] = None,
    custom_generate_next_token: Optional[Callable] = None,
) -> torch.Tensor:
    """
    Generate tokens from a model conditioned on a prompt.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given
            prompt. This is the output of the relevant tokenizer
        max_generated_tokens (int): number of tokens to be generated. This is the max
            since we can stop early based on whether the eos token is respected or not
        temperature (float): value to scale the predicted logits by. Default is 1.0
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within
            the top_k probabilities. Default is None
        stop_tokens (Optional[Set[int]]): If specified, generation is stopped when any of these
            tokens are generated. Default: None
        custom_generate_next_token (Optional[Callable]): If specified, we'll use the custom
            generate_next_token function (e.g. compiled function) when generating the tokens,
            otherwise we'll use the default `geenrate_next_token` function. Default is None

    Returns:
        List: list of generated tokens

    Raises:
        ValueError: if max_seq_len supported by the model is smaller than the number of tokens
            requested
    """
    prompt_length = prompt.size(0)

    if model.max_seq_len < (prompt_length + max_generated_tokens) - 1:
        raise ValueError(
            f"Models maximum seq length {model.max_seq_len} should be >= "
            f"{(prompt_length + max_generated_tokens)} - 1"
        )

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    # generated_tokens is a list of tensors where each tensor contains tokens
    # needed for the output
    generated_tokens = [prompt]

    # generate the first token by conditioning on the input prompt
    token = generate_next_token(
        model=model,
        input_pos=torch.arange(0, prompt_length, device=prompt.device),
        # convert the input into [B, S] shape as expected by the model
        x=prompt.view(1, -1),
        temperature=temperature,
        top_k=top_k,
    ).clone()

    generated_tokens.append(token)

    # generation starts at position=prompt_length and continues till
    # we get the requested number of tokens or we hit eos_id
    input_pos = torch.tensor([prompt_length], device=prompt.device)
    for _ in range(max_generated_tokens - 1):
        token = custom_generate_next_token(
            model=model,
            input_pos=input_pos,
            x=token.view(1, -1),
            temperature=temperature,
            top_k=top_k,
        ).clone()

        if stop_tokens is not None and token.item() in stop_tokens:
            break

        generated_tokens.append(token)

        # update the position before we generate the next token
        input_pos += 1
    return torch.cat(generated_tokens).tolist()
