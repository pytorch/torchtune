# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

# Not ideal to import this type here but it's needed for the transform function
from torchtune.modules import Tokenizer


CROSS_ENTROPY_IGNORE_IDX = -100

_PROMPT_TEMPLATE = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class AlpacaDataset(Dataset):
    """
    Support for the Alpaca dataset and it's variants from HuggingFace Datasets.
    https://huggingface.co/datasets/tatsu-lab/alpaca

    Data input format: https://huggingface.co/datasets/tatsu-lab/alpaca#data-instances

    The input is created using the prompt template from the original alpaca codebase:
    https://github.com/tatsu-lab/stanford_alpaca/blob/761dc5bfbdeeffa89b8bff5d038781a4055f796a/train.py#L31

    This follows the following format:
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:{output}"

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
        use_clean (bool): Whether to use the cleaned version of the dataset or not. Default is False.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
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
        train_on_input: bool = True,
        use_clean: bool = True,
        **kwargs
    ) -> None:
        dataset_path = (
            "yahma/alpaca-cleaned" if use_clean
            else "tatsu-lab/alpaca"
        )
        self._data = load_dataset(dataset_path, split="train")
        self._tokenizer = tokenizer
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self._transform(
            instruction = self._data[index]["instruction"],
            input = self._data[index]["input"],
            output = self._data[index]["output"]
        )

    def _transform(self, instruction: str, input: str, output: str) -> Tuple[List[int], List[int]]:
        """
        Split a sample on ``response`` tag to create input and labels.

        Args:
            sample (str): Sample text.

        Returns:
            Tuple of encoded inputs and labels.
        """
        prompt = self._generate_prompt(instruction, input)
        prompt_with_response = prompt + output

        # add bos always; LlamaTokenizer sets this to True by default and neither
        # alpaca-lora or the original authors change this
        encoded_prompt = self._tokenizer.encode(text=prompt, add_bos=True, add_eos=False)
        encoded_prompt_with_response = self._tokenizer.encode(
            text=prompt_with_response, add_bos=True, add_eos=True
        )
        labels = encoded_prompt_with_response.copy()

        if not self.train_on_input:
            labels[:len(encoded_prompt)] = [CROSS_ENTROPY_IGNORE_IDX] * len(encoded_prompt)

        assert len(encoded_prompt_with_response) == len(labels)
        return encoded_prompt_with_response, labels

    def _generate_prompt(self, instruction: str, input: str) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            instruction (str): Instruction text.
            input (str): Input text.

        Returns:
            Prompt text.
        """
        if input:
            prompt = _PROMPT_TEMPLATE["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            prompt = _PROMPT_TEMPLATE["prompt_no_input"].format(instruction=instruction)
        return prompt
