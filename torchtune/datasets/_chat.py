# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Generator, List, Mapping, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.config._utils import _get_template

from torchtune.data import (
    Dialogue,
    PromptTemplate,
    sharegpt_to_llama2_dialogue,
    tokenize_prompt_and_response,
    truncate,
)
from torchtune.modules import Tokenizer


class ChatDataset(Dataset):
    """
    Class that supports any custom dataset with multiturn conversations.

    The general flow from loading a sample to tokenized prompt is:
    load sample -> apply transform -> foreach turn{format into template -> tokenize}

    If the column/key names differ from the expected names in the `PromptTemplate`,
    then the `column_map` argument can be used to provide this mapping.

    Use `convert_to_dialogue` to prepare your dataset into the llama conversation format
    and roles:
        [
            {
                "role": <system|user|assistant>,
                "content": <message>,
            },
            ...
        ]

    This class supports multi-turn conversations. If a tokenizer sample with multiple
    turns does not fit within `max_seq_len` then it is truncated.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by HuggingFace's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        convert_to_dialogue (Callable[[Mapping[str, Any]], Dialogue]): function that keys into the desired field in the sample
            and converts to a list of `Messages` that follows the llama format with the expected keys
        template (PromptTemplate): template used to format the prompt. If the placeholder variable
            names in the template do not match the column/key names in the dataset, use `column_map` to map them.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        source: str,
        convert_to_dialogue: Callable[[Mapping[str, Any]], Dialogue],
        template: PromptTemplate,
        max_seq_len: int,
        train_on_input: bool = False,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._convert_to_dialogue = convert_to_dialogue
        self.template = template
        self.max_seq_len = max_seq_len
        self.train_on_input = train_on_input

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
        dialogue = self._convert_to_dialogue(sample)

        prompt_tokens = []
        label_tokens = []
        for prompt, label in self._get_turns(dialogue):
            formatted_prompt = self.template.format(prompt)
            encoded_prompt_with_response, labels = tokenize_prompt_and_response(
                tokenizer=self._tokenizer,
                prompt=formatted_prompt,
                response=label,
                train_on_input=self.train_on_input,
            )
            prompt_tokens.extend(encoded_prompt_with_response)
            label_tokens.extend(labels)

            if len(prompt_tokens) >= self.max_seq_len:
                break

        prompt_tokens, label_tokens = truncate(
            self._tokenizer, prompt_tokens, label_tokens, self.max_seq_len
        )

        assert len(prompt_tokens) == len(label_tokens)

        return prompt_tokens, label_tokens

    def _get_turns(
        self, dialogue: Dialogue
    ) -> Generator[Tuple[Dict[str, str], str], None, None]:
        prompt_messages = {}
        for message in dialogue:
            # If we are at the assistant message, we are at the end of a turn, yield.
            if message["role"] == "assistant":
                if "user" not in prompt_messages:
                    raise ValueError(
                        f"Missing a user message before assistant message: {message['content']}"
                    )
                yield prompt_messages, message["content"]
                prompt_messages = {}
            # Otherwise, continue to add to the turn's messages
            else:
                if message["role"] in prompt_messages:
                    raise ValueError(
                        f"Duplicate {message['role']} message in dialogue: {message['content']}"
                    )
                prompt_messages[message["role"]] = message["content"]

        # If we never yielded, then the last turn was incomplete
        if prompt_messages:
            raise ValueError(
                f"Incomplete turn in dialogue, current turn: {prompt_messages}"
            )


def chat_dataset(
    tokenizer: Tokenizer,
    source: str,
    conversation_format: str,
    template: str,
    max_seq_len: int,
    train_on_input: bool = False,
    **load_dataset_kwargs: Dict[str, Any],
) -> ChatDataset:
    """
    Build a configurable dataset with conversations. This method should be
    used to configure a custom chat dataset from the yaml config instead of
    using `ChatDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        source (str): path string of dataset, anything supported by HuggingFace's `load_dataset`
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
        conversation_format (str): string specifying expected format of conversations in the dataset
            for automatic conversion to the llama format. Supported formats are: "sharegpt"
        template (str): class name of template used to format the prompt.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is False.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to `load_dataset`.

    Returns:
        ChatDataset: the configured ChatDataset

    Raises:
        ValueError: if the conversation format is not supported
    """
    if conversation_format == "sharegpt":
        convert_to_dialogue = sharegpt_to_llama2_dialogue
    else:
        raise ValueError(f"Unsupported conversation format: {conversation_format}")

    return ChatDataset(
        tokenizer=tokenizer,
        source=source,
        convert_to_dialogue=convert_to_dialogue,
        template=_get_template(template),
        max_seq_len=max_seq_len,
        train_on_input=train_on_input,
        **load_dataset_kwargs,
    )
