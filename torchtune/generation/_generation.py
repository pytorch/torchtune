# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch
from torchtune.modules.transformer import TransformerDecoder


def multinomial_sample_one(
    probs: torch.Tensor, q: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Samples from a multinomial distribution."""

    q = torch.empty_like(probs).exponential_(1) if q is None else q
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = None,
    q: torch.Tensor = None,
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
    return multinomial_sample_one(probs, q)


def generate_next_token(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    cache_pos: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    q: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next tokens given a prompt, and also returns the corresponding logits.

    Args:
        model (TransformerDecoder): model used for generation
        input_pos (torch.Tensor): tensor with the positional encodings associated with the given prompt,
            with shape [bsz x seq_length].
        x (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape [bsz x seq_length].
        mask (Optional[torch.Tensor]): attention mask with shape [bsz x seq_length x seq_length],
            default None.
        cache_pos (Optional[torch.Tensor]): tensor which contains the cache positions
            of each token, used during inference. This is useful when ``input_ids`` are
            right-shifted to account for padding tokens. Default is None, in which case
            ``input_pos`` is used (if specified).
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): Top-k value to use for sampling, default None.
        q (Optional[torch.Tensor]): randomly sampled tensor for softmax sampling trick.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - tokens (torch.Tensor): tensor with the generated tokens,
                with shape [bsz x 1].
            - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                with shape [bsz x seq_length x vocab_size].

    """
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos, mask=mask, cache_pos=cache_pos)
    return (
        sample(logits[:, -1].clone(), temperature=temperature, top_k=top_k, q=q),
        logits,
    )


def get_causal_mask_from_padding_mask(
    padding_mask: torch.Tensor, target_seq_len: Optional[int] = None
) -> torch.Tensor:
    """
    Converts an attention mask of shape ``[bsz, seq_len]`` to a ``[bsz, seq_len, seq_len]`` causal attention mask suitable for
    consumption by :func:`~torch.nn.functional.scaled_dot_product_attention~`. If ``target_seq_len``
    is provided, this will return a mask of shape ``[bsz, seq_len, target_seq_len]``. This is useful
    when generating masks for static KV caches, e.g. ``target_seq_len=cache_max_seq_len``.

    Args:
        padding_mask (torch.Tensor): Boolean tensor where True indicates the corresponding token in the sequence
            is a padding token and should be masked out in attention, with shape [bsz x seq_length]
        target_seq_len (Optional[int]): target sequence length to create attention mask with. Default None.

    Returns:
        torch.Tensor: Boolean causal mask with shape [bsz, seq_length, seq_length], or
            [bsz, seq_length, target_seq_len] if ``target_seq_len`` was specified.

    Raises:
        AssertionError: if ``target_seq_len > ``seq_len``, the sequence length of the padding mask.

    Example:
        >>> padding_mask = torch.tensor([[True, False, False, False]])
        >>> causal_mask = get_causal_mask_from_padding_mask(padding_mask, target_seq_len=5)
        >>> causal_mask
        >>> torch.Tensor([
        >>>     [True,  False, False, False, False],
        >>>     [False, True,  False, False, False],
        >>>     [False, True,  True,  False, False],
        >>>     [False, True,  True,  True,  False],
        >>>     [False, True,  True,  True,  True]
        >>> ])
    """
    bsz, seq_len = padding_mask.shape
    target_seq_len = seq_len if target_seq_len is None else target_seq_len

    if target_seq_len < seq_len:
        raise AssertionError(
            "target_seq_len cannot be shorter than the sequence length of the padding mask."
        )

    mask = torch.tril(
        torch.ones(seq_len, target_seq_len, device=padding_mask.device, dtype=bool),
        diagonal=0,
    ).repeat(bsz, 1, 1)
    mask.narrow(2, 0, seq_len).mul_(padding_mask[:, None, :].expand(-1, seq_len, -1))
    mask.diagonal(dim1=1, dim2=2).copy_(True)
    return mask


def get_position_ids_from_padding_masks(
    padding_mask: torch.Tensor,
):
    """
    Calculates position ids given a padding mask which right-shifts position ids to start
    from the first valid token.

    Args:
        padding_mask (torch.Tensor): Boolean tensor where True indicates the corresponding token in the sequence
            is a padding token and should be masked out in attention. Shape [bsz, seq_len]

    Returns:
        torch.Tensor: position ids which are appropriately shifted according to any padding values.

    Example:
        >>> padding_mask = torch.tensor([True, True, True, False, False, False, False, False])
        >>> input_pos = get_position_ids_from_padding_mask(padding_mask)
        >>> input_pos
        >>> torch.Tensor([0, 0, 0, 0, 1, 2, 3, 4])
    """
    return ((padding_mask.cumsum(-1) - 1) * padding_mask).to(torch.int)


@torch.inference_mode()
def generate(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    custom_generate_next_token: Optional[Callable] = None,
):
    """
    Generates tokens from a model conditioned on a prompt, and also returns logits for the generations.

    Args:
        model (TransformerDecoder): model used for generation
        prompt (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape either [seq_length] or [bsz x seq_length].
        max_generated_tokens (int): number of tokens to be generated
        pad_id (int): token ID to use for padding, default 0.
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities,
            default None.
        rng (Optional[torch.Generator]): random number generator, default None.
        custom_generate_next_token (Optional[Callable]): If specified, we'll use the
            ``custom_generate_next_token function``. This is generally only useful if
            you want to specify a ``torch.compile`` version of the generate next token for
            performance reasons. If None, we use the default :func:`generate_next_token`.
            Default is None.

    Examples:
        >>> model = torchtune.models.llama3.llama3_8b()
        >>> tokenizer = torchtune.models.llama3.llama3_tokenizer()
        >>> prompt = [0, 0, 0] + tokenizer("Hi my name is") # substitute 0 with pad_id
        >>> rng.manual_seed(42)
        >>> output, logits = generate(model, torch.tensor(prompt), max_generated_tokens=100, pad_id=0)
        >>> print(tokenizer.decode(output[0].tolist()))
        ?? ?? ?? Hi my name is Jeremy and I'm a friendly language model assistant!

    Returns:
        torch.Tensor: Generated tokens.
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    if custom_generate_next_token is None:
        custom_generate_next_token = generate_next_token

    bsz, prompt_length = prompt.size()
    total_response_length = prompt_length + max_generated_tokens

    generated_tokens = prompt.clone()
    incremental_decoding = model.caches_are_enabled()
    max_seq_len = (
        total_response_length if not incremental_decoding else model.cache_max_seq_len
    )
    padding_masks = generated_tokens != pad_id
    cache_pos = None

    if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
            padding_masks, (0, max_generated_tokens), value=True
        )
        masks = get_causal_mask_from_padding_mask(
            padding_masks, target_seq_len=max_seq_len
        )
        input_pos = get_position_ids_from_padding_masks(padding_masks)
        cache_pos = torch.arange(
            0, total_response_length, device=generated_tokens.device
        )
    else:
        masks = torch.tril(
            torch.ones(
                total_response_length,
                max_seq_len,
                dtype=torch.bool,
                device=prompt.device,
            )
        ).unsqueeze(0)
        input_pos = torch.arange(
            0, total_response_length, device=generated_tokens.device
        ).unsqueeze(0)

    del padding_masks

    if incremental_decoding:
        curr_masks = masks[:, :prompt_length]
    else:
        curr_masks = masks[:, :prompt_length, :prompt_length]

    q = torch.empty(
        (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
    ).exponential_(1, generator=rng)
    tokens, _ = generate_next_token(
        model,
        input_pos=input_pos[:, :prompt_length].squeeze(),
        mask=curr_masks,
        cache_pos=cache_pos[:prompt_length] if cache_pos is not None else None,
        x=prompt,
        temperature=temperature,
        top_k=top_k,
        q=q,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    curr_pos = prompt_length
    curr_cache_pos = None
    for _ in range(max_generated_tokens - 1):
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos]
            curr_cache_pos = (
                cache_pos[curr_pos].unsqueeze(0) if cache_pos is not None else None
            )
            curr_masks = masks[:, curr_pos, None, :]
        else:
            tokens = generated_tokens.clone()
            curr_input_pos = input_pos[:, : curr_pos + 1]
            curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        q = torch.empty(
            (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
        ).exponential_(1, generator=rng)
        tokens, logits = custom_generate_next_token(
            model,
            input_pos=curr_input_pos,
            x=tokens,
            cache_pos=curr_cache_pos,
            mask=curr_masks,
            temperature=temperature,
            top_k=top_k,
            q=q,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1

    return generated_tokens, logits
