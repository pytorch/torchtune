# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Optional

from torchtune.data import InputOutputToMessages
from torchtune.data._prompt_templates import PromptTemplate, SummarizeTemplate
from torchtune.datasets._finetune import FinetuneDataset
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.transforms import Transform


def samsum_dataset(
    model_transform: Transform,
    *,
    source: str = "Samsung/samsum",
    column_map: Optional[Dict[str, str]] = None,
    prompt_template: Optional[PromptTemplate] = SummarizeTemplate(),
    train_on_input: bool = False,
    packed: bool = False,
    split: str = "train",
) -> FinetuneDataset:
    """
    Support for summarization datasets and their variants from Hugging Face Datasets.
    An example is the `SAMsum dataset <https://huggingface.co/datasets/samsum>`_.

    The prompt template mirrors what is used in the llama_recipes `codebase
    <https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/datasets/samsum_dataset.py#L13>`_

    where ``dialogue`` and ``summary`` are fields from the dataset.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        model_transform (Transform): model specific transform to convert a list of messages
            output by the dataset to tokens. This will always be a :class:`~torchtune.modules.tokenizers.ModelTokenizer`.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details. Default is ``Samsung/samsum``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns in the prompt template
            to the new column names in the dataset. If None, assume these are identical.
        prompt_template (Optional[PromptTemplate]): optional template used to format the prompt. Default
            is :class:`~torchtune.data.GrammarErrorCorrectionTemplate`.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".

    Returns:
        InstructDataset: dataset configured with source data and template


    Example:
        >>> samsum_ds = samsum_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(samsum_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    column_map = column_map or {"input": "dialogue", "output": "summary"}
    message_transform = InputOutputToMessages(
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
