# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data import (
    CROSS_ENTROPY_IGNORE_IDX,
    InputOutputToMessages,
    InstructTemplate,
    Message,
    validate_messages,
)
from torchtune.data._utils import deprecated
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer


@deprecated(
    msg="Please use `torchtune.datasets.SFTDataset` or :func:`~torchtune.datasets.instruct_dataset` for custom instruct data."
)
class InstructDataset(Dataset):
    """
    Note:
        This class is deprecated and will be removed in a future release. Please use
        :class:`~torchtune.datasets.SFTDataset` or :func:`~torchtune.datasets.instruct_dataset`
        for custom instruct data.

    Class that supports any custom dataset with instruction-based prompts and a
    configurable template.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> format into template -> tokenize

    If the column/key names differ from the expected names in the :class:`~torchtune.data.InstructTemplate`,
    then the ``column_map`` argument can be used to provide this mapping.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``False`` by default.
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        template (InstructTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use ``column_map`` to map them.
        transform (Optional[Callable]): transform to apply to the sample before formatting to the template.
            Default is None.
        column_map (Optional[Dict[str, str]]): a mapping from the expected placeholder names in the template
            to the column/key names in the sample. If None, assume these are identical.
            The output column can be indicated using the ``output`` key mapping.
            If no placeholder for the ``output`` column is provided in ``column_map`` it is assumed to be ``output``.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        max_seq_len (Optional[int]): Maximum number of tokens in the returned input and label token id lists.
            Default is None, disabling truncation. We recommend setting this to the highest you can fit in memory
            and is supported by the model. For example, llama2-7B supports up to 4096 for sequence length.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.
    Raises:
        ValueError: If ``template`` is not an instance of :class:`torchtune.data.InstructTemplate`
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        template: InstructTemplate,
        transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,
        train_on_input: bool = False,
        max_seq_len: Optional[int] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        if not isinstance(template(), InstructTemplate):
            raise ValueError(
                f"template must be an InstructTemplate class, not {type(template())}"
            )

        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.template = template
        self._transform = transform
        self._column_map = column_map
        self.train_on_input = train_on_input
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        transformed_sample = self._transform(sample) if self._transform else sample

        prompt = self.template.format(transformed_sample, self._column_map)
        key_output = (
            self._column_map["output"]
            if self._column_map and "output" in self._column_map
            else "output"
        )
        messages = [
            Message(role="user", content=prompt, masked=(not self.train_on_input)),
            Message(role="assistant", content=transformed_sample[key_output]),
        ]

        validate_messages(messages)

        tokens, mask = self._tokenizer.tokenize_messages(
            messages,
        )

        # Wherever mask == True, set to CROSS_ENTROPY_IGNORE_IDX. Otherwise keep as tokens
        labels = list(np.where(mask, CROSS_ENTROPY_IGNORE_IDX, tokens))
        assert len(tokens) == len(labels)

        return {"tokens": tokens, "labels": labels}


def instruct_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str,
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    packed: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Configure a custom dataset with user instruction prompts and model responses.

    This builder function can be used to configure a custom instruct dataset directly from the yaml config
    as an alternative to :class:`~torchtune.datasets.SFTDataset`, as it is made to be config friendly.

    The dataset should follow this format:

    .. code-block:: text

        |  input          |  output          |
        |-----------------|------------------|
        | "user prompt"   | "model response" |

    If your column names are different, you can use the ``column_map`` parameter to change
    the expected column names. For example, if your dataset has columns ``"question"`` and
    ``"answer"`` you can use::

        column_map = {"input": "question", "output": "answer"}

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
            ``load_dataset`` for more details.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "input"
            and "output" column names to the actual column names in the dataset. Keys should be "input" and
            "output" and values should be the actual column names. Default is None, keeping the default "input"
            and "output" column names.
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is False.
        packed (bool): Whether or not to pack the dataset to tokenizer's ``max_seq_len`` prior to training. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.

    Examples:

    ::

        my_dataset.json
        [
            {
                "question": "What time is it in London?",
                "answer": "It is 10:00 AM in London.",
            },
            {
                ...
            },
            ...,
        ]

    ::

        >>> from torchtune.datasets import instruct_dataset
        >>> dataset = instruct_dataset(
        ...     tokenizer=tokenizer,
        ...     source="json",
        ...     data_files="my_dataset.json",
        ...     column_map={
        ...         "input": "question",
        ...         "output": "answer",
        ...     },
        ...     train_on_input=False,
        ...     packed=False,
        ...     split="train",
        ... )
        >>> tokens = dataset[0]["tokens"]
        >>> tokenizer.decode(tokens)
        "What time is it in London?It is 10:00 AM in London."

    This can also be accomplished via the yaml config:

    .. code-block:: yaml

        dataset:
          _component_: torchtune.datasets.instruct_dataset
          source: json
          data_files: my_dataset.json
          column_map:
            input: question
            output: answer
          train_on_input: False
          packed: False
          split: train

    Returns:
        Union[SFTDataset, PackedDataset]: the configured :class:`~torchtune.datasets.SFTDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.
    """

    message_transform = InputOutputToMessages(
        train_on_input=train_on_input, column_map=column_map
    )

    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
