# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Mapping, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.data import PromptTemplate

from torchtune.modules import Tokenizer


class InstructDataset(Dataset):
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
        **kwargs: Additional keyword arguments to pass to the Alpaca Dataset.


    Example:
        >>> alpaca_ds = AlpacaDataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(alpaca_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        template: Union[PromptTemplate, str],
        text_transform: Optional[Callable] = None,
        token_transform: Optional[Callable] = None,
        column_map: Optional[Dict[str, str]] = None,  # by default, assume column names == what template expects
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source)
        self._template = template
        self._text_transform = text_transform
        self._token_transform = token_transform
        self._column_map = column_map

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._transform(sample)

    def _transform(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        transformed_sample = self._text_transform(sample) if self._text_transform else sample
        prompt = self._template.format(transformed_sample, self._column_map)
        encoded_prompt = self._tokenizer.encode(prompt)
        labels = self._tokenizer.encode(sample[self._column_map["output"]])
        transformed_encoded_prompt, transformed_labels = self._token_transform(encoded_prompt, labels)
        return transformed_encoded_prompt, transformed_labels
