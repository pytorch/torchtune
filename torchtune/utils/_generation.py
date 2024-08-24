# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtune.modules import TransformerDecoder


def multinomial_sample_one(probs: torch.Tensor) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = None
) -> torch.Tensor:
    """Generic sample from a probability distribution."""
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    # change logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_sample_one(probs)


def generate_next_token(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos, mask=mask)[:, -1]
    return sample(logits, temperature, top_k)


def update_stop_tokens_tracker(
    tokens: torch.Tensor, stop_tokens: torch.Tensor, stop_token_reached: torch.Tensor
) -> torch.Tensor:
    """Updates which sequences have reached a stop token."""
    # tokens: [bsz, 1]
    # stop_tokens: [num_stop_tokens]
    # stop_token_reached: [bsz]
    stop_token_reached_curr = torch.isin(tokens, stop_tokens).flatten()
    stop_token_reached |= stop_token_reached_curr
    return stop_token_reached


def pad_left(
    tokens: List[List[int]], *, pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of lists of integers to the left.

    Args:
        tokens (List[List[int]]): token ids
        pad_id (int): padding token id

    Returns:
        Tuple[Tensor, Tensor]: padded tokens and boolean padding mask

    Example:
        >>> tokens = [[1, 2, 3], [5, 6]]
        >>> pad_left(tokens, pad_id = 0)
        (tensor([[1, 2, 3],
                [0, 5, 6]]),
        tensor([[True, True, True],
                [False, True, True]]))
    """
    padded_sequence = pad_sequence(
        [torch.tensor(x[::-1]) for x in tokens],  # first flip each sequence and pad
        batch_first=True,
        padding_value=pad_id,
    ).flip(
        dims=[1]
    )  # flip back to correct order

    padding_mask = padded_sequence != pad_id

    return padded_sequence, padding_mask


def get_causal_mask(
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Converts an attention mask of shape ``[bsz, seq_len]`` to a causal attention mask suitable for
    consumption by :func:`~torch.nn.functional.scaled_dot_product_attention~`.

    HF uses a similar implementation internally, see
    https://github.com/huggingface/transformers/blob/a564d10afe1a78c31934f0492422700f61a0ffc0/src/transformers/models/mistral/modeling_mistral.py#L1096

    Args:
        padding_mask (torch.Tensor): Boolean tensor where True indicates participation in attention
            with shape [bsz x seq_length]

    Returns:
        torch.Tensor: Boolean causal mask with shape [bsz x seq_length x seq_length]

    Example:
        >>> padding_mask = torch.tensor([[1, 1, 1]
        ...                              [1, 1, 0]])
        >>> get_causal_mask(padding_mask)
        tensor([[1, 1, 1],
                [1, 1, 0]]

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
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[List[int]] = None,
    custom_generate_next_token: Optional[Callable] = None,
) -> List[List[int]]:
    """
    Generates tokens from a model conditioned on a prompt.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape either [seq_length] or [bsz x seq_length]
        max_generated_tokens (int): number of tokens to be generated
        mask (Optional[torch.Tensor]): Optional boolean tensor which contains the attention mask
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
        >>> prompt = tokenizer("Hi my name is")
        >>> output = generate(model, prompt, max_generated_tokens=100)
        >>> print(tokenizer.decode(output[0]))
        Hi my name is Jeremy and I'm a friendly language model assistant!

    Returns:
        List[List[int]]: collection of lists of generated tokens
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    # convert stop tokens to tensor for easy matching
    stop_tokens = (
        torch.tensor(stop_tokens, device=prompt.device) if stop_tokens else None
    )
    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()
    # keeps track at a high level if we've already hit a stop token in a sequence so we can early stop
    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    # everything in stop_token_mask starts as 1s, and we'll set them to 0 for sequences
    # that already hit a stop token
    stop_token_mask = torch.ones(
        (bsz, prompt_length + 1), dtype=torch.int32, device=prompt.device
    )

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    # generate the first tokens conditioned on the prompt
    input_pos = torch.arange(0, model.max_seq_len, device=prompt.device)
    tokens = generate_next_token(
        model,
        input_pos=input_pos[:prompt_length],
        x=prompt,
        mask=mask,
        temperature=temperature,
        top_k=top_k,
    )
    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    # stop early if we reach a stop token in every seq
    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(
            tokens, stop_tokens, stop_token_reached
        )
        if stop_token_reached.all().item():
            return generated_tokens.tolist()

    curr_pos = prompt_length
    # if key value caches are enabled, we can incrementally decode
    incremental_decoding = model.caches_are_enabled()
    if incremental_decoding:
        # mask will always just be the smallest causal mask
        mask = torch.ones((bsz, 1, 1), dtype=torch.bool, device=prompt.device)

    for _ in range(max_generated_tokens - 1):
        # update stop_token_mask if we reached a stop token in a previous step
        # by appending the logical not of stop_token_reached to the end of the mask
        # reshaped to be bsz first
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        # if incremental decoding is enabled, we can use the current position
        # otherwise, we take the whole sequence up to the current position
        if incremental_decoding:
            curr_input_pos = input_pos[curr_pos].unsqueeze(0)
        else:
            curr_input_pos = input_pos[: curr_pos + 1]
            tokens = generated_tokens.clone()

            if mask is not None:
                # Add a new row and column to the mask for the new token
                mask = torch.nn.functional.pad(mask, (0, 1, 0, 1))
                # Set the last row to ones (true) to represent causal mask
                mask[:, :, -1] = 1

        tokens = custom_generate_next_token(
            model,
            input_pos=curr_input_pos,
            x=tokens,
            mask=mask,
            temperature=temperature,
            top_k=top_k,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if stop_token_reached.all().item():
                break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        generated_tokens = generated_tokens * stop_token_mask

    return generated_tokens.tolist()
