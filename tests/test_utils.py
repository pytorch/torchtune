# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import functools
import math
import random

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from torch import nn


def set_rng_seed(seed):
    """Sets the seed for random number generators"""
    torch.manual_seed(seed)
    random.seed(seed)


def init_weights_with_constant(model: nn.Module, constant: float = 1.0) -> None:
    for p in model.parameters():
        nn.init.constant_(p, constant)


def fixed_init_tensor(
    shape: torch.Size,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
    dtype: torch.dtype = torch.float,
):
    """
    Utility for generating deterministic tensors of a given shape. In general stuff
    like torch.ones, torch.eye, etc can result in trivial outputs. This utility
    generates a range tensor [min_val, max_val) of a specified dtype, applies
    a sine function if nonlinear=True, then reshapes to the appropriate shape.
    """
    n_elements = math.prod(shape)
    step_size = (max_val - min_val) / n_elements
    x = torch.arange(min_val, max_val, step_size, dtype=dtype)
    x = x.reshape(shape)
    if nonlinear:
        return torch.sin(x)
    return x


@torch.no_grad
def fixed_init_model(
    model: nn.Module,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    nonlinear: bool = False,
):
    """
    This utility initializes all parameters of a model deterministically using the
    function fixed_init_tensor above. See that docstring for details of each parameter.
    """
    for _, param in model.named_parameters():
        param.copy_(
            fixed_init_tensor(
                param.shape,
                min_val=min_val,
                max_val=max_val,
                nonlinear=nonlinear,
                dtype=param.dtype,
            )
        )


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    check_device: bool = True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


from dataclasses import dataclass


@dataclass
class LlamaArgs:
    """
    Dataclass encapsulating various args to instantiate a Llama-2 decoder. The defaults
    are those of a 7b parameter model with a max_seq_len of 2048.

    Args:
        vocab_size (int): Number of entries in vocabulary (default: 32_000)
        embed_dim: (int): Embedding dimension (default: 4096)
        num_layers: (int): Number of Transformer layers (default: 32)
        num_heads (int): Number of attention heads (per layer). (default: 32)
        num_kv_heads: (Optional[int]): Number of key and value heads. This needs to
            be < num_heads and num_heads % num_kv_heads must be 0. `num_kv_heads` can be
            modified to implement GQA or MHA. The default is `None`, in which case
            `num_kv_heads` is set to `num_heads` and MHA is used. Please see
            llm.llama2.attention.LlamaSelfAttention for details.
        max_seq_len: int: Maximum sequence length that this model accepts. Default: 2048
    """

    vocab_size: int = 32_000
    embed_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None
    max_seq_len: int = 2048


def args_7b() -> LlamaArgs:
    return LlamaArgs(
        vocab_size=32_000,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=None,
        max_seq_len=2048,
    )


class LogitsTransform(abc.ABC):
    """Interface for a logits transformation."""

    @abc.abstractmethod
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass


class TemperatureTransform(LogitsTransform):
    """Controls randomness of predicted tokens via a temperature value.

    Args:
        temperature (float): The parameter controlling distribution randomness.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores /= self.temperature
        return scores


class TopPTransform(LogitsTransform):
    """Filters the distribution to cover the fewest tokens whose cumulative mass
    exceeds `prob`.

    Args:
        prob (float): The minimum cumulative probability mass that the kept tokens
            must cover.
    """

    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_sort, scores_index = torch.sort(scores, dim=-1, descending=True)
        scores_cumulative = scores_sort.cumsum(dim=-1)

        # Ignore tokens introducing more probability mass than needed
        discard_mask = scores_cumulative - scores_sort > self.prob
        scores_sort[discard_mask] = 0.0

        scores_sort.div_(scores_sort.sum(dim=-1, keepdim=True))  # renormalize
        scores.scatter_(-1, scores_index, scores_sort)
        return scores


class TopKTransform(LogitsTransform):
    """Filters the distribution to include the top-k highest probability tokens.

    Args:
        top_k (int): The number of highest probability tokens to keep.
    """

    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))
        scores_topk, _ = scores.topk(top_k)

        discard_mask = scores < scores_topk[..., -1]
        scores.masked_fill_(discard_mask, 0.0)

        scores.div_(scores.sum(dim=-1, keepdim=True))  # renormalize
        return scores


def _get_logits_transforms(
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[LogitsTransform]:
    """Returns a list of parameterized logits transforms that can be chained."""
    logits_transforms = []
    if temperature > 0:
        logits_transforms.append(TemperatureTransform(temperature))
    if top_p > 0:
        logits_transforms.append(TopPTransform(top_p))
    if top_k > 0:
        logits_transforms.append(TopKTransform(top_k))

    return logits_transforms


def _apply_logits_transforms(
    logits_transforms: List[LogitsTransform], logits: torch.FloatTensor
) -> torch.FloatTensor:
    """Applies a chained list of logits transforms."""
    output_logits = (
        functools.reduce(lambda x, f: f(x), logits_transforms, logits)
        if logits_transforms
        else logits
    )
    return output_logits


@torch.no_grad()
def generate(
    decoder_lm: torch.nn.Module,
    prompt_tokens: List[List[int]],
    incremental_decode: bool,
    min_gen_len: int,
    max_gen_len: int,
    eos_token_id: int,
    pad_token_id: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 0,
    keep_prompt: bool = True,
    logprobs: bool = False,
    decoder_lm_kwargs: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
    logits_accessor: Optional[Callable] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Interface for generation supporting temperature, top-k, and top-p sampling.

    Args:
        decoder_lm (torch.nn.Module): Input module with which to run `forward`.
        prompt_tokens (List[List[int]]): List of tokenized per-batch prompts.
        min_gen_len (int): Minimum generated sequence length.
        max_gen_len (int): Maximum generated sequence length.
        eos_token_id (int): ID for end-of-sentence token.
        pad_token_id (int): ID for padding token.
        temperature (float): Temperature value to control sampling randomness.
        top_p (float): Probability threshold for nucleus sampling.
        top_k (int): Number of tokens kept for top-k filtering.
        keep_prompt (bool): Whether to keep prompt tokens in the output tensor(s).
        logprobs (bool): Whether to compute log probabilities.
        decoder_lm_kwargs (Optional[Dict[str, Any]]): Additional arguments to pass to `decoder_lm.forward`.
        device (Optional[torch.device]): Device on which to initialize prompt token tensors (should match device of model).
        logits_accessor (Optional[Callable]): Function to extract logits from model output.

    Returns:
        Tuple of generated tokens and optional log probabilities if `logprobs=True`,
        where the dimensions of each tensor are (batch_size, max_gen_length)

    Example:
    ```python
    >>> from transformers import AutoTokenizer, BertModel
    >>> from torchmultimodal.utils.text import generate

    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> model = BertModel.from_pretrained('bert-base-uncased')
    >>> input_prompt = "Today is a good day for"
    >>> prompt_tokens = [tokenizer.encode(input_prompt)]
    >>> tokens, token_logprobs = generate(
    ...     model,
    ...     prompt_tokens,
    ...     min_gen_len=1,
    ...     max_gen_len=10,
    ...     eos_token_id=tokenizer.eos_token_id,
    ...     pad_token_id=tokenizer.sep_token_id,
    ...     logprobs=True
    ... )
    ```
    """
    torch.manual_seed(1337)
    batch_size = len(prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    min_prompt_len = min(len(p) for p in prompt_tokens)
    total_gen_len = max_gen_len + max_prompt_len
    tokens = torch.full(
        (batch_size, total_gen_len), pad_token_id, dtype=torch.long, device=device
    )
    for i, prompt in enumerate(prompt_tokens):
        tokens[i, : len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)
    if logprobs:
        token_logprobs = torch.full_like(
            tokens, float("-inf"), dtype=torch.float, device=device
        )
    else:
        token_logprobs = None
    # mask to ensure we don't overwrite the prompt for prompts > min_prompt_len.
    prompt_mask = tokens != pad_token_id
    logits_transforms = _get_logits_transforms(temperature, top_p, top_k)
    # TODO: generalize the LLM's behavior - for example, models may not take in
    # a start_pos.
    prev_pos = 0
    eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for cur_pos in range(min_prompt_len, total_gen_len):
        input_ids = (
            tokens[:, prev_pos:cur_pos] if incremental_decode else tokens[:, :cur_pos]
        )
        outputs = (
            decoder_lm(input_ids, prev_pos if incremental_decode else 0)
            if decoder_lm_kwargs
            else decoder_lm(input_ids)
        )
        if logits_accessor:
            logits = logits_accessor(outputs)
        else:
            logits = outputs
        next_token_logits = logits[:, -1]

        # Convert to probability distribution, then sample
        next_token_probs = next_token_logits.softmax(dim=-1)
        next_token_probs = _apply_logits_transforms(logits_transforms, next_token_probs)
        next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)
        # Record positions of any EOS tokens across batches
        eos_reached_cur = next_token.eq(eos_token_id)
        eos_reached |= eos_reached_cur
        # Avoid overwriting the prompt for prompts that are longer than min_prompt_len.
        tokens[:, cur_pos] = torch.where(
            prompt_mask[:, cur_pos],
            tokens[:, cur_pos],
            # tokens[:, cur_pos].masked_scatter(~eos_reached, next_token),
            next_token,
        )
        if token_logprobs is not None:
            token_logprobs[:, cur_pos].masked_scatter_(
                ~eos_reached,
                -F.cross_entropy(
                    next_token_logits,
                    tokens[:, cur_pos],
                    reduction="none",
                    ignore_index=pad_token_id,
                ),
            )
        prev_pos = cur_pos
        if eos_reached.all().item():
            break

    if not keep_prompt:
        tokens = tokens[:, max_prompt_len:]
        if token_logprobs is not None:
            token_logprobs = token_logprobs[:, max_prompt_len:]

    return tokens, token_logprobs if logprobs else None
