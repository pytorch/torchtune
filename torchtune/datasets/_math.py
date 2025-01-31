# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import regex
from functools import partial

from typing import Any, Callable, Dict, Optional, Union, TypedDict

from torchtune.data._messages import AlpacaToMessages

from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._rl import RLDataset, ReasoningProblem
from torchtune.modules.tokenizers import ModelTokenizer


def normalize_math(problem: dict[str, str]) -> ReasoningProblem:
    question = problem["problem"]
    REGEXP = r"\\boxed{((?:[^{}]+|\{(?1)\})*)}"
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

