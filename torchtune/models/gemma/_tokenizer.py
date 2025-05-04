# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate, truncate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import (
    ModelTokenizer,
    SentencePieceBaseTokenizer,
)

WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class GemmaTokenizer(ModelTokenizer, Transform):
    """
    Gemma's implementation of the SentencePiece tokenizer

    Args:
        path (str): Path to pretrained tokenizer file.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens. Default is None.
        truncation_type (str): type of truncation to apply, either "left" or "right".
            Default is "right".

    Examples:
        >>> tokenizer = GemmaTokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = None,
        truncation_type: str = "right",
    ):
        self._spm_model = SentencePieceBaseTokenizer(path)

        # Original tokenizer has no pad_id, which causes indexing errors when batch training
        self._spm_model.pad_id = 0

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template
        self.truncation_type = truncation_type

    @property
    def eos_id(self):
        return self._spm_model.eos_id

    @property
    def bos_id(self):
        return self._spm_model.bos_id

    @property
    def pad_id(self):
        return self._spm_model.pad_id

    @property
    def vocab_size(self):
        return self._spm_model.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = False,
    ) -> List[int]:
        return self._spm_model.encode(
            text,
            add_bos=add_bos,
            add_eos=add_eos,
            trim_leading_whitespace=trim_leading_whitespace,
        )

    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        return self._spm_model.decode(token_ids)

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes. Messages should be in prompt-answer format.
            add_end_tokens (bool): Whether to append the EOS token at the end of the
                entire token sequence. This is typically True for training/fine-tuning
                and False for inference/generation. Default is True.

        Returns:
            Tuple[List[int], List[bool]]: A tuple containing the list of token IDs
            and a corresponding list of boolean masks.

        Examples:
            >>> tokenizer = GemmaTokenizer("/path/to/spm_model", max_seq_len=100)
            >>> messages = [
            ...     Message(role="user", content="Hello", masked=True),
            ...     Message(role="assistant", content="World!"),
            ... ]

            >>> # Training/Finetuning (add_end_tokens=True)
            >>> tokens, mask = tokenizer.tokenize_messages(messages, add_end_tokens=True)
            >>> print(tokens) # Example output, specific IDs depend on tokenizer
            [1, 765, 20916, 102]
            >>> print(mask)
            [True, True, False, True]

            >>> # Inference (add_end_tokens=False)
            >>> tokens, mask = tokenizer.tokenize_messages(messages, add_end_tokens=False)
            >>> print(tokens) # Example output, no final EOS
            [1, 765, 20916]
            >>> print(mask)
            [True, True, False]
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]

        for message in templated_messages:
            # Gemma assumes text-only content, extract it
            if isinstance(message.content, list): # Handle potential multi-part format after templating
                text_content = "".join(part["content"] for part in message.content if part["type"] == "text")
            else:
                text_content = message.content

            encoded_tokens = self.encode(
                text_content, add_bos=False, add_eos=False
            )
            tokens.extend(encoded_tokens)
            mask.extend([message.masked] * len(encoded_tokens))

            # Check for max_seq_len after each message to potentially break early
            # Note: This doesn't perfectly guarantee length if a single message exceeds max_seq_len
            if self.max_seq_len and len(tokens) >= self.max_seq_len:
                 break

        if add_end_tokens:
            tokens.append(self.eos_id)
            mask.append(True) # Mask applied to EOS

        if self.max_seq_len:
            tokens = truncate(
                tokens=tokens,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id if add_end_tokens else None,
                truncation_type=self.truncation_type,
            )
            mask = truncate(
                tokens=mask,
                max_seq_len=self.max_seq_len,
                eos_id=True if add_end_tokens else None, # EOS mask is always True
                truncation_type=self.truncation_type,
            )

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
                If True, the final EOS token will not be added. Default is False.

        Returns:
            Mapping[str, Any]: The sample with added "tokens" and "mask" fields
                and the "messages" field removed.
        """
        messages = sample.pop("messages")
        # Pass add_end_tokens based on the inverse of inference flag
        tokens, mask = self.tokenize_messages(messages, add_end_tokens=not inference)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample
