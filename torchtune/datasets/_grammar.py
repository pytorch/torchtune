# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Optional

from torch.utils.data import Dataset
from torchtune.data import ToInputOutputMessages
from torchtune.data._prompt_templates import (
    GrammarErrorCorrectionTemplate,
    PromptTemplate,
)
from torchtune.datasets._finetune import FinetuneDataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms import Transform


def grammar_dataset(
    model_transform: Transform,
    *,
    source: str = "liweili/c4_200m",
    column_map: Optional[Dict[str, str]] = None,
    prompt_template: Optional[PromptTemplate] = GrammarErrorCorrectionTemplate(),
    train_on_input: bool = False,
    packed: bool = False,
    split: str = "train",
) -> Dataset:
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
        source (str): path string of dataset, anything supported by Hugging Face's ``load_dataset``.
        prompt_template (Optional[PromptTemplate]): optional template used to format the prompt. Default
            is :class:`~torchtune.data.GrammarErrorCorrectionTemplate`.
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

    message_transform = ToInputOutputMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    ds = FinetuneDataset(
        source=source,
        message_transform=message_transform,
        model_transform=model_transform,
        prompt_template=prompt_template,
        split=split,
    )
    return PackedDataset(ds) if packed else ds
