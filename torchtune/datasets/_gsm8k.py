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
from torchtune.datasets._sft import SFTDataset

from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._rl import RLDataset, ReasoningProblem
from torchtune.modules.tokenizers import ModelTokenizer

# TODO: dedup this between here and _rl
# SFT_PROMPT =  Assistant: <think>{cot}</think> <answer>{answer}</answer>
PREAMBLE_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: {question} Assistant: """
TRAINABLE_PROMPT = """<think>{cot}</think> <answer>{answer}</answer>"""

def normalize_gsm(problem: dict[str, str]) -> ReasoningProblem:
    question = problem["question"]
    solution = problem["answer"]

    cot, answer = solution.split("#### ")

    return {"question": question, "cot": cot, "answer": answer}


def sft_gsm_transform(problem: dict[str, str]) -> dict[str, str]:
    question = problem["question"]
    solution = problem["answer"]

    cot, answer = solution.split("#### ")

    preamble = PREAMBLE_PROMPT.format(question=question)
    trainable = TRAINABLE_PROMPT.format(cot=cot, answer=answer)

    return {"preamble": preamble, "trainable": trainable}



def gsm8k_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "openai/gsm8k",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "main",
    cheat_idx: int | None = None,
    data_division: int = 1,
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:

    ds = RLDataset(
        source=source,
        name=name,
        tokenizer=tokenizer,
        problem_transform=normalize_gsm,
        filter_fn=filter_fn,
        cheat_idx=cheat_idx,
        data_division=data_division,
        split=split,
        **load_dataset_kwargs,
    )

    return ds


def gsm8k_sft(
    tokenizer: ModelTokenizer,
    *,
    source: str = "openai/gsm8k",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "main",
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:

    def model_transform(problem: dict[str, str]) -> list[int]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False)

        mask = [0 for t in pre_tokens] + [1 for t in trainable_tokens]

        return {"tokens": pre_tokens + trainable_tokens, "mask": mask}


    ds = SFTDataset(
        source=source,
        message_transform=sft_gsm_transform,
        model_transform=model_transform,
        filter_fn=filter_fn,
        split=split,
        name=name,
        **load_dataset_kwargs
    )

    return ds
