# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import AlpacaInstructTemplate
from torchtune.datasets._instruct import InstructDataset
from torchtune.modules import Tokenizer


def alpaca_dataset(
    tokenizer: Tokenizer,
    train_on_input: bool = True,
    max_seq_len: int = 512,
) -> InstructDataset:
    """
    Support for the Alpaca dataset from Hugging Face Datasets.
    https://huggingface.co/datasets/tatsu-lab/alpaca

    Data input format: https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances

    The input is created using the prompt template from the original alpaca codebase:
    https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py#L31

    where `instruction`, `input`, and `output` are fields from the dataset.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `True` by default (ref: https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L49)
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 512, as set by Stanford Alpaca (https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#fine-tuning),
            but we recommend setting this to the highest you can fit in memory and is supported by the model.
            For example, llama2-7B supports up to 4096 for sequence length.

    Returns:
        InstructDataset: dataset configured with Alpaca source data and template


    Example:
        >>> alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return InstructDataset(
        tokenizer=tokenizer,
        source="tatsu-lab/alpaca",
        template=AlpacaInstructTemplate,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split="train",
    )


def alpaca_cleaned_dataset(
    tokenizer: Tokenizer,
    train_on_input: bool = True,
    max_seq_len: int = 512,
) -> InstructDataset:
    """
    Support for the Alpaca cleaned dataset from Hugging Face Datasets.
    https://huggingface.co/datasets/yahma/alpaca-cleaned

    Data input format: https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances

    The input is created using the prompt template from the original alpaca codebase:
    https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py#L31

    where `instruction`, `input`, and `output` are fields from the dataset.

    Masking of the prompt during training is controlled by the `train_on_input` flag, which is
    set to `True` by default (ref: https://github.com/tloen/alpaca-lora/blob/main/finetune.py#L49)
    - If `train_on_input` is True, the prompt is used during training and
    contributes to the loss.
    - If `train_on_input` is False, the prompt is masked out (tokens replaced with -100)

    This is the cleaned version of the original Alpaca dataset, which removes hallucinations,
    poorly formed instructions/inputs/outputs, wrong answers, and other errors. See more details
    on the Hugging Face dataset card.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 512, as set by Stanford Alpaca (https://github.com/tatsu-lab/stanford_alpaca?tab=readme-ov-file#fine-tuning),
            but we recommend setting this to the highest you can fit in memory and is supported by the model.
            For example, llama2-7B supports up to 4096 for sequence length.

    Returns:
        InstructDataset: dataset configured with Alpaca source data and template


    Example:
        >>> alpaca_ds = alpaca_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    return InstructDataset(
        tokenizer=tokenizer,
        source="yahma/alpaca-cleaned",
        template=AlpacaInstructTemplate,
        train_on_input=train_on_input,
        max_seq_len=max_seq_len,
        split="train",
    )
