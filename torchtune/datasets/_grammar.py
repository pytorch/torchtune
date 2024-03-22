# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import GrammarErrorCorrectionTemplate
from torchtune.datasets._instruct import InstructDataset
from torchtune.modules import Tokenizer


def grammar_dataset(
    tokenizer: Tokenizer,
    train_on_input: bool = False,
) -> InstructDataset:
    """
    Support for the Grammar dataset and its variants from HuggingFace Datasets.
    https://huggingface.co/datasets/liweili/c4_200m

    Data input format: https://huggingface.co/datasets/liweili/c4_200m#description

    The prompt template is created from llama_recipes codebase:
    https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset.py#L50

    where `input` and `output` are fields from the dataset.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `False` by default
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.

    Returns:
        InstructDataset: dataset configured with Grammar source data and template


    Example:
        >>> grammar_ds = grammar_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(grammar_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return InstructDataset(
        tokenizer=tokenizer,
        source="liweili/c4_200m",
        template=GrammarErrorCorrectionTemplate(),
        column_map={"sentence": "input"},
        train_on_input=train_on_input,
        split="train",
    )
