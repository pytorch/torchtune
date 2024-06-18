# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchtune.utils._generation import sample, update_stop_tokens_tracker


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
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Converts an attention mask of shape `[bsz, seq_len]` to a causal attention mask suitable for
    consumption by :func:`~torch.nn.functional.scaled_dot_product_attention~`.

    HF uses a similar implementation internally, see
    https://github.com/huggingface/transformers/blob/a564d10afe1a78c31934f0492422700f61a0ffc0/src/transformers/models/mistral/modeling_mistral.py#L1096

    Args:
        padding_mask (torch.Tensor): Boolean tensor where True indicates participation in attention
            with shape [bsz x seq_length]
    Returns:
        torch.Tensor: Boolean causal mask with shape [bsz x seq_length x seq_length]
    """
    _, seq_len = padding_mask.shape
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=padding_mask.device, dtype=bool), diagonal=0
    )
    mask = mask & (padding_mask[:, None, :] & padding_mask[:, :, None])
    mask.diagonal(dim1=1, dim2=2)[:] = True
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

        padding_masks = generated_tokens == pad_id
        if padding_masks.any():
            mask = get_causal_mask(~padding_masks)
            input_pos = (~padding_masks).cumsum(-1) - (~padding_masks).long()
            input_pos = input_pos.to(device=generated_tokens.device, dtype=torch.int)
        else:
            mask = None
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
