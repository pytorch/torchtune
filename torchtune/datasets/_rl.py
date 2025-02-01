# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Mapping, Optional, TypedDict
from xml.etree import ElementTree as ET

import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune import training, utils
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import validate_messages
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


BASE_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"""

class ReasoningProblem(TypedDict):
    question: str
    cot: str
    answer: str

class RLDataset(Dataset):

    def __init__(
        self,
        *,
        source: str,
        problem_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        cheat_idx: int | None = None,
        data_division: int = 1,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._problem_transform = problem_transform
        self._tokenizer = tokenizer
        self._cheat_idx = cheat_idx
        self._data_division = data_division

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data) // self._data_division

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx = index if self._cheat_idx is None else self._cheat_idx
        sample = self._data[idx]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._problem_transform(sample)  # keys "question" and "answer"

        question = BASE_PROMPT % transformed_sample["question"]

        q_tokens = self._tokenizer.encode(question, add_eos=False)
        mask = [1 for _ in q_tokens]
        answer = transformed_sample["answer"]


        return {"tokens": q_tokens, "mask": mask, "answer": answer}



def extract_tags(text: str):
    # Add root element to make valid XML
    xml_string = f"<root>{text}</root>"
    root = ET.fromstring(xml_string)

    return {
        'think': [elem.text for elem in root.findall('think')],
        'answer': [elem.text for elem in root.findall('answer')]
    }


def batch_shaped_correctness_reward(tokenizer: ModelTokenizer, completions: torch.Tensor, answers: list[str]) -> [torch.Tensor, torch.Tensor]:
    batch_size, grpo_size, *_ = completions.shape
    rewards = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    successes = torch.zeros(batch_size, grpo_size, dtype=torch.float32)
    # completions :: [B, G, L]
    for b in range(batch_size):
        for g in range(grpo_size):
            text_completion = tokenizer.decode(completions[b, g].tolist())  # skips special tokens, stops at eos
            reward, success = shaped_correctness_reward(question="", answer=answers[b], completion=text_completion)
            rewards[b, g] = reward
            successes[b, g] = success

    return rewards, successes

def shaped_correctness_reward(question: str, answer: str, completion: str) -> tuple[float, float]:
    question_chars = len(question)
    only_completion = completion[question_chars:]

    reward = 0
    success = 0


    try:
        tags = extract_tags("<think>" + only_completion)
    except ET.ParseError:
        tags = {"think": [], "answer": []}



    if len(tags['answer']) == 1:
        reward += 5.0

    if len(tags['think']) == 1:
        reward += 5.0

    if any(attempt == answer for attempt in tags["answer"]):
        # One of the answer tags has the right answer
        reward += 20.0

    if any(answer in attempt for attempt in tags["answer"]):
        # One of the answer tags contains the right answer (might be e.g. $20 instead of 20
        reward += 10.0

    # total_think_length = sum(len(c) + len("<think></think>") for c in tags["think"])
    # total_answer_length = sum(len(a) + len("<answer></answer>") for a in tags["answer"])
    # total_extra_length = len(only_completion) - total_think_length - total_answer_length

    # reward -= 0.1 * total_extra_length

    if len(tags['answer']) > 0 and tags['answer'][-1] == answer:
        reward = 100.0
        success = 1

    _, rank = utils.get_world_size_and_rank()
    if rank == 0:
        print()
        print(f"{question=}\n{answer=}\n{reward=}\n{only_completion=}")
        print()

    return reward, success


def correctness_reward(question: str, answer: str, completion: str) -> float:
    question_chars = len(question)
    only_completion = completion[question_chars:]

    try:
        tags = extract_tags("<think>" + only_completion)
    except ET.ParseError:
        return 0.0

    if tags['answer'][-1] == answer:
        return 1.0
    else:
        return 0.0


