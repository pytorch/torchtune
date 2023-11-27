# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import abc

import torch


class LogitsTransform(abc.ABC):
    """Interface for a logits transformation."""

    @abc.abstractmethod
    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        pass


class TemperatureTransform(LogitsTransform):
    """Controls randomness of predicted tokens via a temperature value.
    Args:
        temperature (float): The parameter controlling distribution randomness.
    Raises:
        ValueError: If `temperature` is less than or equal to zero.
    """

    def __init__(self, temperature: float):
        if temperature <= 0:
            raise ValueError(f"Expected 0 < `temperature` but got {temperature=}")

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
    Raises:
        ValueError: If `prob` is less than or equal to zero or greater than one.
    """

    def __init__(self, prob: float):
        if prob <= 0 or prob > 1:
            raise ValueError(f"Expected 0 < `prob` <= 1 but got {prob=}")

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
    Raises:
        ValueError: If `top_k` is less than or equal to zero.
        TypeError: If `top_k` is not an integer.
    """

    def __init__(self, top_k: int):
        if top_k <= 0:
            raise ValueError(f"Expected 0 `top_k` > but got {top_k=}")
        if not isinstance(top_k, int):
            raise TypeError(f"Expected `top_k` to be int but got {type(top_k)=}")

        self.top_k = top_k

    def __call__(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))
        scores_topk, _ = scores.topk(top_k)

        discard_mask = scores < scores_topk[..., -1]
        scores.masked_fill_(discard_mask, 0.0)

        scores.div_(scores.sum(dim=-1, keepdim=True))  # renormalize
        return scores
