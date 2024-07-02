# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.datasets._instruct import instruct_dataset, InstructDataset
from torchtune.modules.tokenizers import ModelTokenizer


def grammar_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "liweili/c4_200m",
    train_on_input: bool = False,
    packed: bool = False,
) -> InstructDataset:
    """
    Support for grammar correction datasets and their variants from Hugging Face Datasets.
    Here is an `example <https://huggingface.co/datasets/liweili/c4_200m>`_ of a grammar correction dataset.

    The prompt template mirrors what is used in the `llama_recipes codebase
    <https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/grammar_dataset/grammar_dataset.py#L50>`_

    where ``input`` and ``output`` are fields from the dataset.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.

    Returns:
        InstructDataset: dataset configured with source data and template


    Example:
        >>> grammar_ds = grammar_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(grammar_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return instruct_dataset(
        tokenizer=tokenizer,
        source=source,
        template="torchtune.data.GrammarErrorCorrectionTemplate",
        column_map={"sentence": "input"},
        train_on_input=train_on_input,
        packed=packed,
        split="train",
    )
