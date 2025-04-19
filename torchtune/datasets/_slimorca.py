# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Union

from torchtune.data import ShareGPTToMessages
from torchtune.datasets._packed import PackedDataset

from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def slimorca_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "Open-Orca/SlimOrca-Dedup",
    column_map: Optional[Dict[str, str]] = None,
    masking_strategy: str = "train_on_assistant",
    train_on_input: Optional[bool] = None,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for `SlimOrca-style <https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup>`_
    family of conversational datasets.

    Masking of the prompt during training is controlled by the ``masking_strategy`` parameter which is
    set to ``train_on_assistant`` by default.
    
    - ``train_on_all``: both user and assistant messages are unmasked
    - ``train_on_assistant``: user messages are masked, only assistant messages are unmasked
    - ``train_on_last``: only the last assistant message is unmasked

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text"), pass
            in the filepath in ``data_files``, and set ``split="train"``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``Open-Orca/SlimOrca-Dedup``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns in the message transform
            :class:`~torchtune.data.ShareGPTToMessages` to the new column names in the dataset. Key should
            be "conversations" and value should be the new column name. If None, use
            the default column name ``"conversations"`` in ``Open-Orca/SlimOrca-Dedup``.
        masking_strategy (str): Masking strategy to use for model training.
            Must be one of: ``train_on_all``, ``train_on_assistant``, ``train_on_last``.
            Default is "train_on_assistant".
        train_on_input (bool): Deprecated. Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        packed (bool): Whether or not to pack the dataset to tokenizer's ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with SlimOrca source data

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.

    Example:
        >>> ds = slimorca_dataset(tokenizer=tokenizer)
        >>> for input, label in ds:
        >>>     print(input)
        >>>     print(label)
        >>>
        >>> Sample Output:
        >>> [1, 351, 82, 391, 221, 220, 193, 12, 471, ..., 2]
        >>> [-100, -100, -100, -100, -100, -100, -100, -100, 471, ..., 2]
    """
    message_transform = ShareGPTToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
        masking_strategy=masking_strategy,
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
