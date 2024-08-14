# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Optional, Union, Tuple

from torchtune.data import InputOutputToMessages, Role
from torchtune.data._prompt_templates import PromptTemplate, SummarizeTemplate
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer


def samsum_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "Samsung/samsum",
    column_map: Optional[Dict[str, str]] = None,
    prompt_template: Optional[Dict[Role, Tuple[str, str]]] = None,
    train_on_input: bool = False,
    packed: bool = False,
    split: str = "train",
) -> Union[SFTDataset, PackedDataset]:
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
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``Samsung/samsum``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns "input" and "output"
            to the new column names in the dataset. If None, use ``{"input": "dialogue", "output", "summary"}``.
        prompt_template (Optional[Dict[Role, Tuple[str, str]]]): optional template used to format the prompt. If None
            is specified, :class:`~torchtune.data.SummarizeTemplate` will be used.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to tokenizer's ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and template

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.

    Example:
        >>> samsum_ds = samsum_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(samsum_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """
    column_map = column_map or {"input": "dialogue", "output": "summary"}

    if prompt_template is None:
        template = SummarizeTemplate()
    else:
        template = PromptTemplate(template=prompt_template)

    message_transform = InputOutputToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        prompt_template=template,
        split=split,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
