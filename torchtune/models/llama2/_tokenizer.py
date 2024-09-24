# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping, Optional, Tuple

from torchtune.data import Message, PromptTemplate
from torchtune.models.llama2._prompt_template import Llama2ChatTemplate
from torchtune.modules.tokenizers import (
    ModelTokenizer,
    SentencePieceBaseTokenizer,
    tokenize_messages_no_special_tokens,
)
from torchtune.modules.transforms import Transform

WHITESPACE_CHARS = [" ", "\n", "\t", "\r", "\v"]


class Llama2Tokenizer(ModelTokenizer, Transform):
    """
    Llama2's implementation of the SentencePiece tokenizer. Llama2Tokenizer does
    not include any additional special tokens. The prompt template described in
    https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2/ describes
    [INST][/INST] and <<SYS>><</SYS>> as special tokens but these are not registered
    as unique ids and are tokenized as normal text. When using this tokenizer on the
    pre-trained model for inference, the prompt template
    :class:`~torchtune.models.llama2.Llama2ChatTemplate` is by default applied to your data
    before tokenization to add the [INST] and <<SYS>> tags for optimal performance.
    For more details, see https://pytorch.org/torchtune/main/tutorials/chat.html#tokenizing-prompt-templates-special-tokens.

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
        >>> tokenizer = Llama2Tokenizer("/path/to/spm_model")
        >>> tokenized_text = tokenizer.encode("Hello world!", add_bos=True, add_eos=True)
        >>> print(tokenized_text)
        [1, 31587, 29644, 102, 2]
    """

    def __init__(
        self,
        path: str,
        max_seq_len: Optional[int] = None,
        prompt_template: Optional[PromptTemplate] = Llama2ChatTemplate(),
    ):
        self._spm_model = SentencePieceBaseTokenizer(path)

        # Original tokenizer has no pad_id, which causes indexing errors when batch training
        self._spm_model.pad_id = 0

        # During generation, stop when eos_id is encountered
        self.stop_tokens = [self.eos_id]

        self.max_seq_len = max_seq_len

        self.prompt_template = prompt_template

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
        return tokenize_messages_no_special_tokens(
            tokenizer=self,
            messages=templated_messages,
            bos_id=self.bos_id if add_start_tokens else None,
            eos_id=self.eos_id if add_end_tokens else None,
        )

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
