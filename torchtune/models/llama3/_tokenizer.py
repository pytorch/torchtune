# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.tokenizers import ModelTokenizer, TikTokenBaseTokenizer
from torchtune.modules.transforms import Transform


CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa

SPECIAL_TOKENS = {
    "<|begin_of_text|>": 128000,
    "<|end_of_text|>": 128001,
    "<|reserved_special_token_0|>": 128002,
    "<|reserved_special_token_1|>": 128003,
    "<|finetune_right_pad_id|>": 128004,
    "<|step_id|>": 128005,
    "<|start_header_id|>": 128006,
    "<|end_header_id|>": 128007,
    "<|eom_id|>": 128008,
    "<|eot_id|>": 128009,
    "<|python_tag|>": 128010,
    "<|image|>": 128256,
    "<|video|>": 128012,
}

NUM_RESERVED_SPECIAL_TOKENS = 256

RESERVED_TOKENS = {
    f"<|reserved_special_token_{2 + i}|>": 128013 + i
    for i in range(NUM_RESERVED_SPECIAL_TOKENS - len(SPECIAL_TOKENS))
}

LLAMA3_SPECIAL_TOKENS = {**SPECIAL_TOKENS, **RESERVED_TOKENS}


class Llama3Tokenizer(ModelTokenizer, Transform):
    """
    tiktoken tokenizer configured with Llama3 Instruct's special tokens, as described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        special_tokens (Optional[Dict[str, int]]): mapping containing special text tokens and
            their registered token IDs. If left as None, this will be set to the canonical
            Llama3 special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.

    Examples:
        >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.special_tokens = (
            special_tokens if special_tokens is not None else LLAMA3_SPECIAL_TOKENS
        )

        self._validate_special_tokens()

        # Encode BOS and EOS, define pad ID
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = self.special_tokens["<|finetune_right_pad_id|>"]
        self.step_id = self.special_tokens["<|step_id|>"]

        # Encode extra special tokens
        self.start_header_id = self.special_tokens["<|start_header_id|>"]
        self.end_header_id = self.special_tokens["<|end_header_id|>"]
        self.eot_id = self.special_tokens["<|eot_id|>"]

        self.eom_id = self.special_tokens["<|eom_id|>"]
        self.python_tag = self.special_tokens["<|python_tag|>"]

        # Media tokens
        self.image_id = self.special_tokens["<|image|>"]

        # During generation, stop when either eos_id, eot_id, or eom_id is encountered
        self.stop_tokens = [self.eos_id, self.eot_id, self.eom_id]

        self.tt_model = TikTokenBaseTokenizer(
            path=path,
            name="llama3_tiktoken",
            pattern=CL100K_PATTERN,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=self.special_tokens,
        )
        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

        # Regex for removing special tokens from the decoded string
        self._special_token_regex = re.compile(r"<\|.*?\|>")
        self._special_token_header_regex = re.compile(
            r"<\|start_header_id\|>.*?<\|end_header_id\|>\n\n"
        )

    def _validate_special_tokens(
        self,
    ):
        """
        Validate that required special tokens are passed into the tokenizer.
        """
        for token in [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ]:
            if token not in self.special_tokens:
                raise ValueError(f"{token} missing from special_tokens")

    def _remove_special_tokens(self, text: str) -> str:
        """
        Remove special tokens from the decoded string.
        """
        # First remove the headers, then the remaining special tokens
        return self._special_token_regex.sub(
            "", self._special_token_header_regex.sub("", text)
        )

    @property
    def base_vocab_size(self) -> int:
        return self.tt_model.base_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.tt_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        return self.tt_model.encode(text=text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: List[int],
        truncate_at_eos: bool = True,
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode a list of token ids into a string.

        Args:
            token_ids (List[int]): The list of token ids.
            truncate_at_eos (bool): Whether to truncate the string at the end of
                sequence token. Default is True.
            skip_special_tokens (bool): Whether to show or skip special tokens in the decoded string.
                Default is True.

        Returns:
            str: The decoded string.
        """
        # We will remove special tokens manually via regex on the decoded string.
        # This is because removing all special tokens does not remove the role and
        # whitespace added from the special tokens, i.e., the "user" and "\n\n" in
        # "<|start_header_id|>user<|end_header_id|>\n\n"
        decoded_string = self.tt_model.decode(
            token_ids=token_ids,
            truncate_at_eos=truncate_at_eos,
        )
        return (
            self._remove_special_tokens(decoded_string)
            if skip_special_tokens
            else decoded_string
        )

    def _tokenize_header(self, message: Message) -> List[int]:
        """
        Tokenize header start, message role, and header end as list of ids
        """
        return (
            [self.start_header_id]
            + self.encode(message.role.strip(), add_bos=False, add_eos=False)
            + [self.end_header_id]
            + self.encode("\n\n", add_bos=False, add_eos=False)
        )

    def _tokenize_end(self, message: Message) -> List[int]:
        """
        Add eot or eom id at the end of the message.
        """
        return [self.eot_id] if message.eot else [self.eom_id]

    def _tokenize_body(self, message: Message) -> List[int]:
        """
        Tokenize message content as list of ids
        """
        tokenized_body = []
        for item in message.content:
            if item["type"] == "text":
                tokenized_body += self.encode(
                    item["content"].strip(), add_bos=False, add_eos=False
                )
            elif item["type"] == "image":
                tokenized_body += [self.image_id]
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")

        if message.ipython:
            tokenized_body = [self.python_tag] + tokenized_body

        return tokenized_body

    def tokenize_message(
        self,
        message: Message,
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_start_tokens (bool): Whether to prepend a tokenized header to the message. Default is True.
            add_end_tokens (bool): Whether to append eot or eom id at the end of the message. Default is True.

        Returns:
            List[int]: The list of token ids.
        """
        tokenized_header = self._tokenize_header(message) if add_start_tokens else []
        tokenized_body = self._tokenize_body(message)
        tokenized_end = self._tokenize_end(message) if add_end_tokens else []

        tokenized_message = tokenized_header + tokenized_body + tokenized_end
        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into a list of token ids and masks.

        Args:
            messages (List[Message]): The list of messages to tokenize.
            add_end_tokens (bool): Whether to append end tokens ids (end-of-seq, end-of-turn, end-of-message) at the end of the
                last assistant message. This value should be set to False for generation. Default is True.

        Examples:
            >>> # Tokenize a list of messages with default settings
            >>> messages = [
            ...     Message(role="user", content="Hello world!", masked=True),
            ...     Message(role="assistant", content="How are you?", masked=False),
            ... ]
            >>> tokenizer = Llama3Tokenizer("/path/to/tt_model")
            >>> tokenizer.tokenize_messages(messages)
            ([1, 31587, 29644, 102, 1, 31587, 29644, 102, 2], [True, True, True, True, True, False, False, False, True])

            >>> # Tokenize a list of messages with add_end_tokens set to False
            >>> tokenizer.tokenize_messages(messages, add_end_tokens=False)
            ([1, 31587, 29644, 102, 1, 31587, 29644], [True, True, True, True, True, False, False])

        Returns:
            Tuple[List[int], List[bool]]: The list of token ids and the list of masks.
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]

        num_messages = len(templated_messages)
        for i, message in enumerate(templated_messages):
            # Add end tokens to the last assistant message if add_end_tokens is True
            # Otherwise, end tokens should always be added
            add_end_tokens_to_message = (
                add_end_tokens if i == num_messages - 1 else True
            )
            tokenized_message = self.tokenize_message(
                message, add_end_tokens=add_end_tokens_to_message
            )

            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if self.max_seq_len and len(tokens) >= self.max_seq_len:
                break

        if add_end_tokens:
            tokens = tokens + [self.eos_id]
            mask = mask + [True]

        if self.max_seq_len:
            tokens = truncate(
                tokens, self.max_seq_len, self.eos_id if add_end_tokens else None
            )
            mask = truncate(mask, self.max_seq_len, True if add_end_tokens else None)

        return tokens, mask

    def __call__(
        self, sample: Mapping[str, Any], inference: bool = False
    ) -> Mapping[str, Any]:
        """
        Apply ``tokenize_messages`` to the "messages" field in the sample.

        Args:
            sample (Mapping[str, Any]): A sample with a "messages" field containing
                a List[Message] to tokenize
            inference (bool): Whether the template is being used for inference or not.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
