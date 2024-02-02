# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchtune.modules import Tokenizer, TransformerDecoder
from torchtune.utils.logits_transforms import (
    LogitsTransform,
    TemperatureTransform,
    TopKTransform,
    TopPTransform,
)


class GenerationUtils:
    """Utility class for generating text from a decoder-style LLM.

    Args:
        decoder_lm (nn.Module): Transformer-Decoder based language model.
        pad_id (int): Padding token ID.
        eos_id (int): End-of-sequence token ID.

    NOTE:
        Currently, `decoder_lm` assumes a forward API with the signature
        `def forward(x: torch.Tensor, curr_pos: int)` as the index of the
        current token is passed in for kv-caching during incremental decoding.
        If `decoder_lm` does not support this interface, please set
        `incremental_decode` to `False` when calling `generate` function.
    """

    def __init__(self, decoder_lm: nn.Module, pad_id: int, eos_id: int):
        self.decoder_lm = decoder_lm
        self.pad_id = pad_id
        self.eos_id = eos_id

    def _get_logits_transforms(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> List[LogitsTransform]:
        """Returns a list of parameterized logits transforms that can be chained.

        Args:
            temperature (float): Sampling temperature.
            top_p (float): Probability threshold for nucleus sampling.
            top_k (int): Number of tokens kept for top-k filtering.

        Returns:
            List of LogitsTransform objects.
        """
        logits_transforms = []
        if temperature > 0:
            logits_transforms.append(TemperatureTransform(temperature))
        if top_p > 0:
            logits_transforms.append(TopPTransform(top_p))
        if top_k > 1:
            logits_transforms.append(TopKTransform(top_k))

        return logits_transforms

    def _apply_logits_transforms(
        self, logits_transforms: List[LogitsTransform], logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Applies a chained list of logits transforms.

        Args:
            logits_transforms (List[LogitsTransform]): List of LogitsTransform objects.
            logits (torch.FloatTensor): Raw logits tensor.

        Returns:
            Transformed logits tensor.
        """
        output_logits = (
            functools.reduce(lambda x, f: f(x), logits_transforms, logits)
            if logits_transforms
            else logits
        )
        return output_logits

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        min_gen_len: int,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 1,
        keep_prompt: bool = True,
        logprobs: bool = False,
        incremental_decode: bool = True,
        logits_accessor: Optional[Callable] = None,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Interface for generation supporting temperature, top-k, and top-p sampling.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized per-batch prompts.
            min_gen_len (int): Minimum generated sequence length.
            max_gen_len (int): Maximum generated sequence length.
            temperature (float): Temperature value to control sampling randomness. Defaults to 0.6.
            top_p (float): Probability threshold for nucleus sampling. Defaults to 0.9.
            top_k (int): Number of tokens kept for top-k filtering. Defaults to 1.
            keep_prompt (bool): Whether to keep prompt tokens in the output tensor(s). Defaults to True.
            logprobs (bool): Whether to compute log probabilities. Defaults to False.
            incremental_decode (bool): Whether to decode incrementally or not. Defaults to True.
            logits_accessor (Optional[Callable]): Function to transform logits before sampling. Defaults to None.
            device (Optional[torch.device]): Device on which to initialize prompt token tensors (should match device of model).
                Defaults to torch.device("cpu").

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Tuple of generated tokens and optional log probabilities if `logprobs=True`,
            where the dimensions of each tensor are (batch_size, max_gen_length)

        Example:
            >>> LLaMA = GenerationUtils(model, pad_id = tokenizer.pad_id, eos_id = tokenizer.eos_id)
            >>> tokens = LLaMA.generate(
            ...     [tokenizer.encode(["I love to eat"])],
            ...     min_gen_len=5,
            ...     max_gen_len=20,
            ...     temperature=0.8,
            ...     top_p=0.7,
            ...     keep_prompt=True,
            ... )
            >>> print(tokens)
            ["I love to eat ice cream"]
        """
        torch.manual_seed(1337)

        batch_size = len(prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        min_prompt_len = min(len(p) for p in prompt_tokens)
        total_gen_len = max_gen_len + max_prompt_len
        tokens = torch.full(
            (batch_size, total_gen_len), self.pad_id, dtype=torch.long, device=device
        )
        for i, prompt in enumerate(prompt_tokens):
            tokens[i, : len(prompt)] = torch.tensor(
                prompt, dtype=torch.long, device=device
            )
        if logprobs:
            token_logprobs = torch.full_like(
                tokens, float("-inf"), dtype=torch.float, device=device
            )
        else:
            token_logprobs = None
        # mask to ensure we don't overwrite the prompt for prompts > min_prompt_len.
        prompt_mask = tokens != self.pad_id
        logits_transforms = self._get_logits_transforms(temperature, top_p, top_k)
        # TODO: generalize the LLM's behavior - for example, models may not take in
        # a start_pos.
        prev_pos = 0
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for cur_pos in range(min_prompt_len, total_gen_len):
            input_ids = tokens[:, prev_pos:cur_pos]
            if incremental_decode:
                outputs = self.decoder_lm(input_ids, curr_pos=prev_pos)
            else:
                outputs = self.decoder_lm(input_ids)
            if logits_accessor:
                logits = logits_accessor(outputs)
            else:
                logits = outputs
            next_token_logits = logits[:, -1]

            # Convert to probability distribution, then sample
            next_token_probs = next_token_logits.softmax(dim=-1)
            next_token_probs = self._apply_logits_transforms(
                logits_transforms, next_token_probs
            )
            next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
            # Record positions of any EOS tokens across batches
            eos_reached_cur = next_token.eq(self.eos_id)
            eos_reached |= eos_reached_cur
            # Avoid overwriting the prompt for prompts that are longer than min_prompt_len.
            tokens[:, cur_pos] = torch.where(
                prompt_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_token,
            )
            if token_logprobs is not None:
                token_logprobs[:, cur_pos].masked_scatter_(
                    ~eos_reached,
                    -F.cross_entropy(
                        next_token_logits,
                        tokens[:, cur_pos],
                        reduction="none",
                        ignore_index=self.pad_id,
                    ),
                )

            if incremental_decode:
                prev_pos = cur_pos

            if eos_reached.all().item():
                break

        if not keep_prompt:
            tokens = tokens[:, max_prompt_len:]
            if token_logprobs is not None:
                token_logprobs = token_logprobs[:, max_prompt_len:]

        return tokens, token_logprobs if logprobs else None


def generate_from_prompt(
    prompt: str, tokenizer: Tokenizer, decoder: TransformerDecoder
) -> Tuple[str, List[int]]:
    """
    Generate a response from a prompt and a decoder.
    Args:
        prompt (str): Prompt to generate from.
        tokenizer (Tokenizer): Tokenizer to use for generation.
        decoder (TransformerDecoder): Model to use for generation.

    Returns:
        Tuple[str, List[int]]: Generated response and corresponding tokenized response.
    """
    prompt_tokens = [tokenizer.encode(prompt, add_eos=False)]
    with torch.no_grad():
        generations_no_kv_cache, _ = GenerationUtils(
            decoder_lm=decoder,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
        ).generate(
            prompt_tokens=prompt_tokens,
            incremental_decode=False,
            min_gen_len=1,
            max_gen_len=256,
            top_k=3,
            device=torch.cuda.current_device(),
        )
    gens = generations_no_kv_cache.tolist()[0]
    gen_str = tokenizer.decode(gens)
    return gens, gen_str
