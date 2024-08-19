# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Union

from torchtune.data import ShareGPTToMessages
from torchtune.datasets._packed import PackedDataset

from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms import Transform


def slimorca_dataset(
    model_transform: Transform,
    *,
    source: str = "Open-Orca/SlimOrca-Dedup",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    split: str = "train",
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for `SlimOrca-style <https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup>`_
    family of conversational datasets.

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
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``Open-Orca/SlimOrca-Dedup``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns in the message transform
            :class:`~torchtune.data.ShareGPTToMessages` to the new column names in the dataset. If None, use
            the default column name ``"conversations"`` in ``Open-Orca/SlimOrca-Dedup``.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        new_system_prompt (Optional[str]): if specified, prepend a system message to every sample. This can
            serve as instructions to guide the model response. Setting this will OVERRIDE any system
            messages already present in the dataset. Default is None.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with SlimOrca source data

    Example:
        >>> ds = slimorca_dataset(model_transform=tokenizer)
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
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=model_transform,
        split=split,
    )
    return PackedDataset(ds) if packed else ds
