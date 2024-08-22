# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from torchtune.modules.transformer import TransformerDecoder


def multinomial_sample_one(probs: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
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
    mask = torch.tril(torch.ones(seq_len, seq_len, device=padding_mask.device, dtype=bool), diagonal=0)
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
    custom_generate_next_token_with_logits=None,
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
    # if custom_generate_next_token is None:
    if custom_generate_next_token_with_logits is None:
        custom_generate_next_token_with_logits = generate_next_token_with_logits

    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()

    padding_masks = generated_tokens == pad_id

    if padding_masks.any():
        # todo just create max length input pos and index in
        prompt_masks = get_causal_mask(~padding_masks)
        masks = torch.tril(
            torch.ones(
                model.max_seq_len,
                model.max_seq_len,
                dtype=torch.bool,
                device=prompt.device,
            ).repeat(bsz, 1, 1)
        )
        masks[:, :prompt_length, :prompt_length] = prompt_masks
        masks[:, prompt_length:, :prompt_length] = prompt_masks[:, -1:, :]

        input_pos = (~padding_masks).cumsum(-1) - (~padding_masks).long()
        input_pos = input_pos.to(torch.int)
        start_positions = input_pos.max(dim=-1, keepdim=False)[0]
        extended_input_pos = torch.stack(
            [
                torch.arange(start_pos + 1, start_pos + max_generated_tokens, dtype=torch.int, device=prompt.device)
                for start_pos in start_positions
            ]
        )
        input_pos = torch.hstack((input_pos, extended_input_pos))
    else:
        masks = None
        curr_masks = None
        input_pos = torch.arange(0, prompt_length + max_generated_tokens, device=generated_tokens.device).view(1, -1)

    incremental_decoding = model.caches_are_enabled()

    if incremental_decoding:
        if masks is not None:
            model.causal_mask = None
            curr_masks = masks[:, :prompt_length]
        curr_pos = input_pos[:, :prompt_length].squeeze()
    else:
        if masks is not None:
            curr_masks = masks[:, :prompt_length, :prompt_length]
        curr_pos = input_pos[:, :prompt_length]
        # curr_masks = None

    import pdb

    # pdb.set_trace()
    # lets grab the first tokens
    _, tokens = generate_next_token_with_logits(
        model,
        input_pos=curr_pos,
        mask=curr_masks,
        x=prompt,
        temperature=temperature,
        top_k=top_k,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    curr_pos = prompt_length

    for _ in range(max_generated_tokens - 1):
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos]
            if curr_masks is not None:
                # input_pos_expanded = curr_input_pos.unsqueeze(1).expand(-1, masks.shape[-1])
                # import pdb

                # pdb.set_trace()
                # # Use advanced indexing to select rows from the mask
                # curr_masks = torch.gather(
                #     masks, 1, curr_input_pos.unsqueeze(1).expand(-1, masks.shape[-1], -1).to(torch.long)
                # )
                curr_masks = masks[torch.arange(bsz), curr_input_pos].unsqueeze(1)
                # curr_masks = masks[:, curr_input_pos, curr_input_pos]
        else:
            curr_input_pos = input_pos[:, : curr_pos + 1]
            tokens = generated_tokens.clone()
            if curr_masks is not None:
                curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        # pdb.set_trace()
        logits, tokens = custom_generate_next_token_with_logits(
            model,
            input_pos=curr_input_pos.unsqueeze(-1),
            x=tokens,
            mask=curr_masks,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1
        # if padding_masks.any():
        #     mask = torch.nn.functional.pad(mask, (0, 1, 0, 1))
        #     mask[:, -1, :] = mask[:, -2, :]
        #     mask[..., -1, -1] = True

    return generated_tokens, logits
