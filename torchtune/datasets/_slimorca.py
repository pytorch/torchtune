# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from torchtune.datasets._common import CROSS_ENTROPY_IGNORE_IDX

# Not ideal to import this type here but it's needed for the transform function
from torchtune.modules import Tokenizer


class _Llama2ChatFormatConstants:
    """
    Contains constants that are used in Llama2 Chat Format.
    """

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class SlimOrcaDataset(Dataset):
    """
    PyTorch Representation of the SlimOrca Dataset
    https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup
    from Hugging Face.

    The data is formatted to adhere to Llama2 Chat Format.
    This format is required if the base model is Llama2 Chat Model.
    The base Llama2 Model doesn't prescribe a particular format.

    The returned data is a tuple of input token id list and label token id
    list. If `max_token_length` keyword argument is provided, the returned
    input token id list is ensured (by truncation if necessary) to be within
    that length.

    Data input format: https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup#dataset-format

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        **kwargs: Additional keyword arguments to pass to the SlimOrca Dataset.

    Keyword Arguments:
        max_token_length (int): Maximum number of tokens in the returned input and label token id lists. This value needs to be at least 4 though it is generally set to max sequence length accepted by the model. Default is 1024.

    Raises:
        ValueError: If `max_token_length` is less than 4.

    Example:
        >>> ds = SlimOrcaDataset(tokenizer=tokenizer, max_token_length=10)
        >>> for input, label in ds:
        >>>     print(input)
        >>>     print(label)
        >>>
        >>> Sample Ouput:
        >>> [1, 351, 82, 391, 221, 220, 193, 12, 471, ..., 2]
        >>> [-100, -100, -100, -100, -100, -100, -100, -100, 471, ..., 2]
    """  # noqa

    def __init__(self, tokenizer: Tokenizer, **kwargs) -> None:
        self._data = load_dataset("Open-Orca/SlimOrca-Dedup", split="train")
        self._tokenizer = tokenizer
        self._max_token_length = kwargs.get("max_token_length", 1024)
        if self._max_token_length < 4:
            # Input token needs to have 1 bos, 1 eos,
            # and 1 token from prompt, 1 from label
            raise ValueError("max_token_length must be at least 4")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        data = self._data[index]["conversations"]
        prompt, label = self._generate_prompt_label(data)
        return self._generate_tokens(prompt, label)

    def _generate_tokens(self, prompt: str, label: str) -> Tuple[List[int], List[int]]:
        """
        Given a prompt string and label string, generate input and label token id lists.

        Tokenizer is used to tokenize both the strings.
        The prompt token list is truncated to `max_token_length` - 2
        (so that there is at least one label token, as EOS takes one token).

        The label token list is truncated to `max_token_length` - len(prompt_token_list)

        Finally input token list is the concatenation of prompt and label token lists.

        Label token list is padded with cross entropy ignore idx value to match the length of input token list.
        """
        prompt_tokens = self._tokenizer.encode(prompt, add_bos=True, add_eos=False)
        # Truncate to max token length - 2 (so that there is at least one label token)
        prompt_tokens = prompt_tokens[: self._max_token_length - 2]

        # Calculate space left for label tokens
        label_tokens_length = self._max_token_length - len(prompt_tokens)
        label_tokens = self._tokenizer.encode(label, add_bos=False, add_eos=True)

        # Truncate label tokens
        label_tokens = label_tokens[: label_tokens_length - 1]
        if label_tokens[-1] != self._tokenizer.eos_id:
            label_tokens.append(self._tokenizer.eos_id)

        input = prompt_tokens + label_tokens
        label = [
            CROSS_ENTROPY_IGNORE_IDX for _ in range(len(prompt_tokens))
        ] + label_tokens
        return input, label

    def _generate_prompt_label(self, data: List[Dict[str, str]]) -> Tuple[str, str]:
        """
        Construct prompt and label strings adhering to Llama2 Chat Format.
        This method supports only back-and-forth conversation per sample (as it is sufficient for SlimOrca dataset).
        """
        agent_text_dict = {}
        # agents can be {system, human, gpt}
        for conversation in data:
            agent = conversation["from"]
            text = conversation["value"]
            agent_text_dict[agent] = text

        # Llama2 Chat Format - https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
        if "system" in agent_text_dict:
            prompt = f"{_Llama2ChatFormatConstants.B_INST} {_Llama2ChatFormatConstants.B_SYS}{agent_text_dict['system']}{_Llama2ChatFormatConstants.E_SYS}{agent_text_dict['human']} {_Llama2ChatFormatConstants.E_INST}"  # noqa: B950
        else:
            prompt = f"{_Llama2ChatFormatConstants.B_INST} {agent_text_dict['human']} {_Llama2ChatFormatConstants.E_INST}"

        response = f" {agent_text_dict['gpt']} "
        return prompt, response
