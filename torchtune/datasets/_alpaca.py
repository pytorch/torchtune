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
    use_clean: bool = False,
) -> InstructDataset:
    """
    Support for the Alpaca dataset and its variants from HuggingFace Datasets.
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

    The version of the dataset used is controlled by the `use_clean` flag which set to False by default.
    - If `use_clean` is True, then https://huggingface.co/datasets/yahma/alpaca-cleaned is used
    - If `use_clean` is False, then https://huggingface.co/datasets/tatsu-lab/alpaca is used

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        use_clean (bool): Whether to use the cleaned version of the dataset or not. Default is False.

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
        source="yahma/alpaca-cleaned" if use_clean else "tatsu-lab/alpaca",
        template=AlpacaInstructTemplate(),
        train_on_input=train_on_input,
        split="train",
    )
