# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional

import regex

from torchtune.datasets._rl import ReasoningProblem, RLDataset
from torchtune.modules.tokenizers import ModelTokenizer

# Attempts to extract the last boxed expression from the solution, which should be the final answer..
REGEXP = r"\\boxed{((?:[^{}]+|\{(?1)\})*)}"


def normalize_math(problem: dict[str, str]) -> ReasoningProblem:
    question = problem["problem"]
    answer = regex.findall(REGEXP, problem["solution"])[-1]

    return {"question": question, "cot": "", "answer": answer}


def math_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "EleutherAI/hendrycks_math",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "algebra",
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:

    ds = RLDataset(
        source=source,
        name=name,
        tokenizer=tokenizer,
        problem_transform=normalize_math,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    return ds
