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


def normalize_math(problem: dict[str, str]) -> ReasoningProblem:
    """
    Parses an item from the DeepScaleR dataset into a ReasoningProblem by transforming it into the question, cot, and answer.
    """
    return {
        "question": problem["problem"],
        "cot": problem["solution"],
        "answer": problem["answer"],
    }


def sft_math_transform(problem: dict[str, str]) -> dict[str, str]:
    """
    Prepares an item from the MATH into a format that can be used for SFT.
    """
    question = problem["problem"]
    cot = problem["solution"]
    answer = re.search(r"\\boxed\{(.*?)\}", cot)

    preamble = PREAMBLE_PROMPT.format(question=question)
    trainable = TRAINABLE_PROMPT.format(cot=cot, answer=answer)

    return { "preamble": preamble, "trainable": trainable}


def math_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "EleutherAI/hendrycks_math",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "algebra",
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> RLDataset:
    """
    MATH dataset from EleutherAI, prepared for RL-based training with verifiable rewards.
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
        problem_transform=normalize_math,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        **load_dataset_kwargs,
    )

    return ds


def math_sft(
    tokenizer: ModelTokenizer,
    *,
    source: str = "EleutherAI/hendrycks_math",
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    name: str = "algebra",
    partition: Optional[str] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> SFTDataset:
    """
    MATH dataset from EleutherAI, prepared for SFT-based training with CoT.
    """

    def model_transform(problem: dict[str, str]) -> dict[str, list[int]]:
        pre_tokens = tokenizer.encode(problem["preamble"], add_eos=False)
        trainable_tokens = tokenizer.encode(problem["trainable"], add_bos=False)

        # 1 == discard the token, 0 == include the token in training
        mask = [1 for _ in pre_tokens] + [0 for _ in trainable_tokens]

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
        message_transform=sft_math_transform,
        model_transform=model_transform,
        filter_fn=filter_fn,
        filter_kwargs=dict(with_indices=True),
        split=split,
        name=name,
        **load_dataset_kwargs,
    )

    return ds
