# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import Llama2ChatFormat, sharegpt_to_llama2_messages

from torchtune.datasets._chat import ChatDataset

from torchtune.modules.tokenizers import Tokenizer


def slimorca_dataset(
    tokenizer: Tokenizer,
    source: str = "Open-Orca/SlimOrca-Dedup",
    max_seq_len: int = 1024,
    train_on_input: bool = False,
) -> ChatDataset:
    """
    Support for `SlimOrca-style <https://huggingface.co/datasets/Open-Orca/SlimOrca-Dedup>`_
    family of conversational datasets.

    The data is formatted to adhere to Llama2 Chat Format.
    This format is required if the base model is Llama2 Chat Model.
    The base Llama2 Model doesn't prescribe a particular format.

    The returned data is a tuple of input token id list and label token id
    list. If `max_seq_len` keyword argument is provided, the returned
    input token id list is ensured (by truncation if necessary) to be within
    that length.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            This value needs to be at least 4 though it is generally set to max sequence length accepted by the model.
            Default is 1024.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.

    Raises:
        ValueError: If `max_seq_len` is less than 4.

    Returns:
        ChatDataset: dataset configured with SlimOrca source data and LLaMA2 chat template

    Example:
        >>> ds = slimorca_dataset(tokenizer=tokenizer, max_seq_len=10)
        >>> for input, label in ds:
        >>>     print(input)
        >>>     print(label)
        >>>
        >>> Sample Output:
        >>> [1, 351, 82, 391, 221, 220, 193, 12, 471, ..., 2]
        >>> [-100, -100, -100, -100, -100, -100, -100, -100, 471, ..., 2]
    """
    if max_seq_len < 4:
        # Input token needs to have 1 bos, 1 eos,
        # and 1 token from prompt, 1 from label
        raise ValueError("max_seq_len must be at least 4")

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_messages=sharegpt_to_llama2_messages,
        chat_format=Llama2ChatFormat,
        max_seq_len=max_seq_len,
        train_on_input=train_on_input,
        split="train",
    )
