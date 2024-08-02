# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torchtune.modules.transformer import TransformerDecoder


def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
    rng: Optional[torch.Generator] = None,
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
    return multinomial_sample_one(probs, rng)


def generate_next_token_with_logits(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
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
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): Top-k value to use for sampling, default None.
        rng (Optional[torch.Generator]): random number generator, default None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                with shape [bsz x seq_length x vocab_size].
            - tokens (torch.Tensor): tensor with the generated tokens,
                with shape [bsz x 1].

    """
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos, mask=mask)
    return logits, sample(logits[:, -1].clone(), temperature, top_k, rng)


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
    """
    _, seq_len = padding_mask.shape
    mask = torch.tril(
        torch.ones(seq_len, seq_len, device=padding_mask.device, dtype=bool), diagonal=0
    )
    mask = mask & (padding_mask[:, None, :] & padding_mask[:, :, None])
    mask.diagonal(dim1=1, dim2=2)[:] = True
    return mask


@torch.inference_mode()
def generate_with_logits(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
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

    Examples:
        >>> model = torchtune.models.llama3.llama3_8b()
        >>> tokenizer = torchtune.models.llama3.llama3_tokenizer()
        >>> prompt = [0, 0, 0] + tokenizer("Hi my name is") # substitute 0 with pad_id
        >>> rng = torch.Generator() # optionally place on device
        >>> rng.manual_seed(42)
        >>> output = generate(model, torch.tensor(prompt), max_generated_tokens=100, pad_id=0, rng=rng)
        >>> print(tokenizer.decode(output[0]))
        ?? ?? ?? Hi my name is Jeremy and I'm a friendly language model assistant!

    Returns:
        torch.Tensor: Generated tokens.
    """
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    _, prompt_length = prompt.size()
    generated_tokens = prompt.clone()

    for i in range(max_generated_tokens):
        padding_masks = generated_tokens == pad_id
        if padding_masks.any():
            mask = get_causal_mask(~padding_masks)
            input_pos = (~padding_masks).cumsum(-1) - (~padding_masks).long()
            input_pos = input_pos.to(torch.int)
        else:
            mask = None
            input_pos = torch.arange(
                0, prompt_length + i, device=generated_tokens.device
            )

        logits, tokens = generate_next_token_with_logits(
            model,
            input_pos=input_pos,
            x=generated_tokens,
            mask=mask,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    return generated_tokens, logits
