# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import regex
from xml.etree import ElementTree as ET
from functools import partial

from typing import Any, Callable, Dict, Optional, Union, TypedDict

from torchtune.data._messages import AlpacaToMessages

from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._rl import RLDataset, ReasoningProblem
from torchtune.modules.tokenizers import ModelTokenizer



def normalize_gsm(problem: dict[str, str]) -> ReasoningProblem:
    question = problem["question"]
    solution = problem["answer"]

    cot, answer = solution.split("#### ")

    return {"question": question, "cot": cot, "answer": answer}


def gsm8k_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "openai/gsm8k",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "main",
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:

    ds = RLDataset(
        source=source,
        name=name,
        tokenizer=tokenizer,
        problem_transform=normalize_gsm,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )

    return ds

