# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

from transformers import LlamaTokenizer

from torchtune.data import Message, PromptTemplate
from torchtune.models.sarvam1._prompt_template import Sarvam1ChatTemplate
from torchtune.models.sarvam1._utils import tokenize_messages_no_special_tokens
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform


class Sarvam1Tokenizer(ModelTokenizer, Transform):
    """
    This is the same as the Llama2Tokenizer, but with special handling for the spiece_underline token.

    Args:
        path (str): Path to pretrained SentencePiece tokenizer file.
        max_seq_len (Optional[int]): A max sequence length to truncate tokens to.
            Default: None
        prompt_template (Optional[PromptTemplate]): template used to format the messages based on their role. This is used
            to add structured text around the actual messages. The structured text is used in three scenarios:

            - Task-specific templates to gear models for a particular task that it will expect after training
            - Model-specific templates that are required whenever the model is prompted, such as the [INST]
              tags in Llama2 and in Mistral
            - Community standardized templates, such as :class:`~torchtune.data.ChatMLTemplate`

            The extra text will still get tokenized as normal text, not as special tokens.
            Default is :class:`~torchtune.models.llama2.Llama2ChatTemplate`.

    Examples:
        >>> tokenizer = Sarvam1Tokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = Sarvam1ChatTemplate(),
    ):
        if not path.endswith(".model"):
            raise ValueError(f"Tokenizer path must end with '.model', got {path}")

        self._tokenizer = LlamaTokenizer(vocab_file=path, legacy=False)
        self._tokenizer.pad_id = self.eos_id if self.pad_id is None else self.pad_id

        self.stop_tokens = [self.eos_id]

        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

    @property
    def eos_id(self):
        return self._tokenizer.eos_token_id

    @property
    def bos_id(self):
        return self._tokenizer.bos_token_id

    @property
    def pad_id(self):
        return self._tokenizer.pad_token_id

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        trim_leading_whitespace: bool = True,
    ) -> List[int]:
        """
        Encode a string into a list of tokens.
        Note:
            Currently this method does not add eos token, and does not trim leading whitespace.
        """

        if trim_leading_whitespace:
            # newline is token so it can be used to trim leading whitespace
            prefix = "\n"
            encoded_prefix = self._tokenizer.encode(prefix, add_special_tokens=False)
            start_idx = len(encoded_prefix)
            tokens = self._tokenizer.encode(prefix + text, add_special_tokens=False)[
                start_idx:
            ]
        else:
            tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if add_bos:
            tokens = [self.bos_id] + tokens

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(
        self,
        token_ids: List[int],
    ) -> str:
        return self._tokenizer.decode(token_ids)

    def tokenize_messages(
        self,
        messages: List[Message],
        *,
        add_start_tokens: bool = True,
        add_end_tokens: bool = True,
    ) -> Tuple[List[int], List[bool]]:
        r"""Tokenize a list of messages one at a time then concatenate them,
        returning a list of tokens and a list of masks.

        Note:
            sentencepiece has problems where in general
            encode(s1 + s2) != encode(s1) + encode(s2) due to whitespace handling.
            We can get around this by prepending s2 with a known token and slicing the
            beginning off the tokenized s2.

        Example:
            >>> tokenizer = Llama2Tokenizer(tokenizer_path, max_seq_len)
            >>> messages = [
                Message(role="system", content="system message\n", masked=True),
                Message(role="user", content="user prompt\n", masked=True),
                Message(role="assistant", content="assistant response\n"),
            ]

            >>> # tokenize_messages encodes messages separately and concats
            >>> tokenizer.tokenize_messages(messages)[0]
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]

            >>> # Same result as encoding the full string in one go
            >>> tokenizer.encode(''.join([message.content for message in messages]))
            [1, 1788, 2643, 13, 1792, 9508, 13, 465, 22137, 2933, 2]


        Args:
            messages (List[Message]): A list of messages, each containing role, content,
                and masked attributes.
            add_start_tokens (bool): Whether to add BOS token to the beginning of the first message.
                Default True.
            add_end_tokens (bool): Whether to add EOS token to the end of the last message. Default True.

        Returns:
            Tuple[List[int], List[bool]]: The tokenized messages
        """
        templated_messages = (
            self.prompt_template(messages)
            if self.prompt_template is not None
            else messages
        )

        tokenized_messages = tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=templated_messages,
            bos_id=self.bos_id if add_start_tokens else None,
            eos_id=self.eos_id if add_end_tokens else None,
        )

        return tokenized_messages

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
