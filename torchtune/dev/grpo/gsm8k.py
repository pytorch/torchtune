# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import re
from typing import Any, Callable, Dict, Optional

from torchtune.datasets import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer

from .data import ReasoningProblem, RLDataset

# TODO: dedup this between here and _rl
PREAMBLE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: {question} Assistant: "
)

TRAINABLE_PROMPT = "<think>{cot}</think> <answer>{answer}</answer>"


def normalize_gsm(problem: dict[str, str]) -> ReasoningProblem:
    """
    Parses an item from the GSM8K dataset into a ReasoningProblem by splitting it up into the question, cot, and answer.
    """
    question = problem["question"]
    solution = problem["answer"]

    cot, answer = solution.split("#### ")

    return {"question": question, "cot": cot, "answer": answer}


def sft_gsm_transform(problem: dict[str, str]) -> dict[str, str]:
    """
    Prepares an item from the GSM8k into a format that can be used for SFT.
    """
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
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:
    """
    GSM8k dataset from OpenAI, prepared for RL-based training with verifiable rewards.
    """

    def default_filter_fn(example: dict, idx: int):
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())

        current = idx % total
        return start <= current <= end

    filter_fn = filter_fn if filter_fn is not None else default_filter_fn

    ds = RLDataset(
        source=source,
        name=name,
        tokenizer=tokenizer,
        problem_transform=normalize_gsm,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
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
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    GSM8k dataset from OpenAI, prepared for SFT-based training with CoT.
    """

    def model_transform(problem: dict[str, str]) -> dict[str, list[int]]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False)

        # 1 == discard the token, 0 == include the token in training
        mask = [1 for t in pre_tokens] + [0 for t in trainable_tokens]

        return {"tokens": pre_tokens + trainable_tokens, "mask": mask}

    def default_filter_fn(example: dict, idx: int):
        if partition is None:
            return True

        match = re.match(r"^(\d+)-(\d+)/(\d+)$", partition)
        if not match:
            raise ValueError(
                f"Invalid partition format: {partition}. Expected format: start-end/total"
            )

        start, end, total = map(int, match.groups())

        current = idx % total
        return start <= current <= end

    filter_fn = filter_fn if filter_fn is not None else default_filter_fn

    ds = SFTDataset(
        source=source,
        message_transform=sft_gsm_transform,
        model_transform=model_transform,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        name=name,
        **load_dataset_kwargs,
    )

    return ds
