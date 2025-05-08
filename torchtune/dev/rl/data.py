# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, List, Mapping, Optional, TypedDict, Union

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from torchtune.data import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

BASE_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, "
    "i.e., <think>reasoning process here</think> <answer>answer here</answer>. User: %s. Assistant: <think>"
)


class ReasoningProblem(TypedDict):
    question: str
    cot: str
    answer: str


class RLDataset(Dataset):
    """
    Base class for datasets used in reinforcement learning,
    which provide a reference answer that can be verified to compute rewards.
    """

    def __init__(
        self,
        *,
        source: str,
        problem_transform: Transform,
        tokenizer: ModelTokenizer,
        filter_fn: Optional[Callable] = None,
        filter_kwargs: Optional[dict[str, Any]] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._problem_transform = problem_transform
        self._tokenizer = tokenizer

        self._data = load_dataset(source, **load_dataset_kwargs)
        if filter_fn is not None:
            if filter_kwargs is None:
                filter_kwargs = {}
            self._data = self._data.filter(filter_fn, **filter_kwargs)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        transformed_sample = self._problem_transform(
            sample
        )  # keys "question" and "answer"

        question = BASE_PROMPT % transformed_sample["question"]

        q_tokens = self._tokenizer.encode(question, add_eos=False)
        mask = [1 for _ in q_tokens]
        answer = transformed_sample["answer"]

        return {
            "question": question,
            "tokens": q_tokens,
            "mask": mask,
            "answer": answer,
        }


def padded_collate_rl(
    batch: List[Dict[str, List[int]]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors. Answers are simply concatenated into a list.

    Args:
        batch (List[Dict[str, List[int]]]): A list of dictionaries containing tokens.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.

    Returns:
        Dict[str, Union[torch.Tensor, List[str]]]: Collated input tensors and string answers.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "answer": "15"},
        >>>    {"tokens": [7,], "answer": "bromance"},
        >>> ]
        >>> collated = padded_collate_rl(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["answers"]
        >>> ["15", "bromance"]
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )

    answers = [x["answer"] for x in batch]
    text = [x["question"] for x in batch]

    return {"tokens": input_ids.long(), "answers": answers, "text": text}
