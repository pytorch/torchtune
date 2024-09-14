# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from typing import Any, Dict, Mapping, Optional, Union

from torchtune.data import Message
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class AlpacaToMessages(Transform):
    """
    Message transform class for Alpaca-style datasets with "instruction", "input", and "output"
    (or equivalent fields specified in column_map) columns. User messages are formed from the
    instruction + input columns and assistant messages are formed from the output column. Prompt
    templating is conditional on the presence of the "input" column, and thus is handled directly
    in this transform class instead of a dedicated :class:`~torchtune.data.PromptTemplate` class
    due to this custom logic.

    Args:
        train_on_input (bool): Whether the model is trained on the user prompt or not.
            Default is True.
        column_map (Optional[Dict[str, str]]): a mapping to change the expected "instruction", "input",
            and "output" column names to the actual column names in the dataset. Default is None,
            keeping the default column names.
    """

    def __init__(
        self, train_on_input: bool = True, column_map: Optional[Dict[str, str]] = None
    ):
        self.train_on_input = train_on_input
        self.column_map = column_map
        self.template = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
        }

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        column_map = self.column_map or {}
        key_input = column_map.get("input", "input")
        key_instruction = column_map.get("instruction", "instruction")
        key_output = column_map.get("output", "output")

        if key_input in sample and sample[key_input]:
            prompt = self.template["prompt_input"].format(
                instruction=sample[key_instruction], input=sample[key_input]
            )
        else:
            prompt = self.template["prompt_no_input"].format(
                instruction=sample[key_instruction]
            )

        messages = [
            Message(
                role="user",
                content=prompt,
                masked=not self.train_on_input,
                eot=True,
            ),
            Message(
                role="assistant",
                content=sample[key_output],
                masked=False,
                eot=True,
            ),
        ]
        return {"messages": messages}


def alpaca_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "tatsu-lab/alpaca",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = True,
    packed: bool = False,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for family of Alpaca-style datasets from Hugging Face Datasets using
    the `data input format <https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances>`_
    and `prompt template <https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L31>`_
    from the original alpaca codebase, where ``instruction``, ``input``, and ``output``
    are fields from the dataset. This template is automatically applied independent
    of any prompt template configured in the tokenizer.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``True`` by `default <https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L49>`_
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``tatsu-lab/alpaca``.
        column_map (Optional[Dict[str, str]]): a mapping from the expected columns in the message transform
            :class:`~torchtune.data.AlpacaToMessages` to the new column names in the dataset. Keys should be
            "instruction", "input", and "output" and values should be the actual column names. If None, uses
            the default column names ``"instruction``, ``"input"``, and ``"output"`` in ``tatsu-lab/alpaca``.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not set on the tokenizer.

    Example:
        >>> alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = AlpacaToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
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


alpaca_cleaned_dataset = partial(alpaca_dataset, source="yahma/alpaca-cleaned")
alpaca_cleaned_dataset.__doc__ = """
Builder for a variant of Alpaca-style datasets with the cleaned version of the
original Alpaca dataset, `yahma/alpaca-cleaned <https://huggingface.co/datasets/yahma/alpaca-cleaned>`_.
See the dataset page and :func:`~torchtune.datasets.alpaca_dataset` for more details.
"""
