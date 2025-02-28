# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
    TikTokenBaseTokenizer,
)

PATTERN = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa

SPECIAL_TOKENS_MAP = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "[INST]": 3,
    "[/INST]": 4,
    "[AVAILABLE_TOOLS]": 5,
    "[/AVAILABLE_TOOLS]": 6,
    "[TOOL_RESULTS]": 7,
    "[/TOOL_RESULTS]": 8,
    "[TOOL_CALLS]": 9,
    "[IMG]": 10,
    "<pad>": 11,
    "[IMG_BREAK]": 12,
    "[IMG_END]": 13,
    "[PREFIX]": 14,
    "[MIDDLE]": 15,
    "[SUFFIX]": 16,
    "[SYSTEM_PROMPT]": 17,
    "[/SYSTEM_PROMPT]": 18,
    "[TOOL_CONTENT]": 19,
}

# Add reserved special tokens
NUM_RESERVED_SPECIAL_TOKENS = 999
RESERVED_TOKENS = {
    f"<SPECIAL_{20 + i}>": 20 + i
    for i in range(NUM_RESERVED_SPECIAL_TOKENS - len(SPECIAL_TOKENS_MAP))
}

SPECIAL_TOKENS_MAP = {**SPECIAL_TOKENS_MAP, **RESERVED_TOKENS}


class MistralSmallTokenizer(ModelTokenizer, Transform):
    """Tokenizer for Mistral Small model using TikToken implementation.

    Args:
        path (str): Path to pretrained tiktoken tokenizer file.
        special_tokens (Optional[Dict[str, Dict]]): mapping containing special text tokens and
            their properties. If left as None, this will be set to the canonical
            Mistral special tokens.
        max_seq_len (Optional[int]): maximum sequence length for tokenizing a single list of messages,
            after which the input will be truncated. Default is None.
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role.
    """

    def __init__(
        self,
        path: str,
        special_tokens: Optional[Dict[str, Dict]] = None,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        self.special_tokens = (
            special_tokens if special_tokens is not None else SPECIAL_TOKENS_MAP
        )

        self._validate_special_tokens()

        # Encode special tokens
        self.bos_id = self.special_tokens["<s>"]["id"]
        self.eos_id = self.special_tokens["</s>"]["id"]
        self.pad_id = self.special_tokens["<pad>"]["id"]
        self.unk_id = self.special_tokens["<unk>"]["id"]

        # Tool and instruction tokens
        self.inst_start_id = self.special_tokens["[INST]"]["id"]
        self.inst_end_id = self.special_tokens["[/INST]"]["id"]
        self.tool_content_id = self.special_tokens["[TOOL_CONTENT]"]["id"]

        # Media tokens
        self.img_id = self.special_tokens["[IMG]"]["id"]
        self.img_break_id = self.special_tokens["[IMG_BREAK]"]["id"]
        self.img_end_id = self.special_tokens["[IMG_END]"]["id"]

        # During generation, stop at eos
        self.stop_tokens = [self.eos_id]

        # Initialize the base tokenizer
        special_tokens_map = {
            token: info["id"] for token, info in self.special_tokens.items()
        }

        self.tt_model = TikTokenBaseTokenizer(
            path=path,
            name="mistral_small",
            pattern=PATTERN,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            special_tokens=special_tokens_map,
        )

        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

        # Build regex pattern from special tokens
        special_tokens_pattern = "|".join(
            [
                re.escape(token)
                for token in self.special_tokens.keys()
                if not token.startswith(
                    "<SPECIAL_"
                )  # Handle reserved tokens separately
            ]
        )
        reserved_tokens_pattern = r"<SPECIAL_\d+>"  # Pattern for reserved tokens

        # Combine patterns with word boundaries to ensure exact matches
        self._special_token_regex = re.compile(
            rf"\b({special_tokens_pattern}|{reserved_tokens_pattern})\b"
        )

    def _validate_special_tokens(self):
        """Validate that required special tokens are passed into the tokenizer."""
        for token in [
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "[INST]",
            "[/INST]",
            "[TOOL_CONTENT]",
            "[IMG]",
            "[IMG_BREAK]",
            "[IMG_END]",
        ]:
            if token not in self.special_tokens:
                raise ValueError(f"{token} missing from special_tokens")

    def _remove_special_tokens(self, text: str) -> str:
        """Remove special tokens from the decoded string."""
        return self._special_token_regex.sub("", text)

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
            truncate_at_eos (bool): Whether to truncate at the end of sequence token.
            skip_special_tokens (bool): Whether to show or skip special tokens.

        Returns:
            str: The decoded string.
        """
        decoded_string = self.tt_model.decode(
            token_ids=token_ids,
            truncate_at_eos=truncate_at_eos,
        )
        return (
            self._remove_special_tokens(decoded_string)
            if skip_special_tokens
            else decoded_string
        )

    def _tokenize_message(
        self,
        message: Message,
        *,
        add_inst_tokens: bool = True,
    ) -> List[int]:
        """
        Tokenize a message into a list of token ids.

        Args:
            message (Message): The message to tokenize.
            add_inst_tokens (bool): Whether to add instruction tokens.

        Returns:
            List[int]: The list of token ids.

        Raises:
            RuntimeError: If an unsupported message content type is encountered.
        """
        tokenized_message = []

        if add_inst_tokens:
            tokenized_message.append(self.inst_start_id)

        for item in message.content:
            if item["type"] == "text":
                tokenized_message.extend(
                    self.encode(item["content"].strip(), add_bos=False, add_eos=False)
                )
            elif item["type"] == "image":
                tokenized_message.extend(
                    [self.img_id, self.img_break_id, self.img_end_id]
                )
            else:
                raise RuntimeError(f"Unsupported message content type: {item['type']}")

        if add_inst_tokens:
            tokenized_message.append(self.inst_end_id)

        return tokenized_message

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        """
        Tokenize a list of messages into token ids and masks.

        Args:
            messages (List[Message]): The messages to tokenize.
            add_end_tokens (bool): Whether to add end tokens.

        Returns:
            Tuple[List[int], List[bool]]: Token ids and masks.
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokens = [self.bos_id]
        mask = [True]  # bos is always masked

        for message in templated_messages:
            tokenized_message = self._tokenize_message(message)
            tokens.extend(tokenized_message)
            mask.extend([message.masked] * len(tokenized_message))

            if self.max_seq_len and len(tokens) >= self.max_seq_len:
                break

        if add_end_tokens:
            tokens.append(self.eos_id)
            mask.append(True)

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
        Apply tokenize_messages to the sample's messages field.

        Args:
            sample (Mapping[str, Any]): Sample with messages field.
            inference (bool): Whether in inference mode.

        Returns:
            Mapping[str, Any]: Sample with tokens and mask fields.
        """
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
